"""
LangGraph agent for the Procurement AI Assistant.

Adds CONVERSATION MEMORY and QUERY REWRITING on top of the RAG chain.

Why rewriting matters: a follow-up like "And what is the maximum cap on that?"
means nothing to the vector search ("that" is not in any document). So we first
ask the LLM to turn it into a standalone question using the chat history
("What is the maximum cap on the Acme late-delivery penalty?"), and search with that.

The graph (one run per user message):

    START -> rewrite -> retrieve -> generate -> remember -> END

  rewrite   : follow-up question -> standalone question (skipped on the first message)
  retrieve  : FAISS search with the standalone question
  generate  : answer from the retrieved chunks, with [n] citations
  remember  : append this exchange to the chat history

Memory: a LangGraph checkpointer (MemorySaver) stores the state per `session_id`
(called thread_id in LangGraph). Same session_id = same conversation.
It lives in the server's RAM, so it resets when the server restarts.

Demo:  python -m src.agent.graph
"""

from typing import Dict, List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.config import settings
from src.rag.chain import _get_index, build_sources, format_context
from src.rag.llm import get_llm
from src.rag.prompts import RAG_PROMPT
from src.vectorstore.faiss_store import similarity_search

MAX_HISTORY_MESSAGES = 8  # keep the last 4 exchanges (user + assistant each)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You rewrite follow-up questions. Given a chat history and the user's "
            "latest question, rewrite the latest question so it is fully "
            "self-contained: replace pronouns and vague references (it, that, they, "
            "the contract, ...) with what they refer to. If the question is already "
            "self-contained, return it unchanged. Output ONLY the rewritten "
            "question, nothing else.",
        ),
        (
            "human",
            "Chat history:\n{history}\n\nLatest question: {question}\n\nStandalone question:",
        ),
    ]
)


class AgentState(TypedDict, total=False):
    question: str              # what the user just typed
    k: Optional[int]           # how many chunks to retrieve
    history: List[Dict]        # [{"role": "user"|"assistant", "content": str}, ...]
    standalone_question: str   # rewritten, self-contained question
    docs: List[Dict]           # retrieved chunks as plain dicts (page_content, metadata)
    answer: str
    sources: List[Dict]


# ---------------------------------- nodes ----------------------------------
def rewrite_node(state: AgentState) -> Dict:
    question = state["question"]
    history = state.get("history", [])
    if not history:
        return {"standalone_question": question}

    history_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}"
        for m in history[-MAX_HISTORY_MESSAGES:]
    )
    rewritten = (REWRITE_PROMPT | get_llm() | StrOutputParser()).invoke(
        {"history": history_text, "question": question}
    )
    rewritten = rewritten.strip().strip('"').strip()
    return {"standalone_question": rewritten or question}


def retrieve_node(state: AgentState) -> Dict:
    docs = similarity_search(
        _get_index(),
        state["standalone_question"],
        k=state.get("k") or settings.top_k_results,
    )
    # Stored as plain dicts so the checkpointer can save them safely
    return {"docs": [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]}


def generate_node(state: AgentState) -> Dict:
    docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in state["docs"]]
    answer = (RAG_PROMPT | get_llm() | StrOutputParser()).invoke(
        {"context": format_context(docs), "question": state["standalone_question"]}
    )
    return {"answer": answer.strip(), "sources": build_sources(docs, answer)}


def remember_node(state: AgentState) -> Dict:
    history = state.get("history", []) + [
        {"role": "user", "content": state["question"]},
        {"role": "assistant", "content": state["answer"]},
    ]
    return {"history": history[-MAX_HISTORY_MESSAGES:]}


# ---------------------------------- graph ----------------------------------
_graph = None


def get_graph():
    """Build the graph once and reuse it (the memory lives inside it)."""
    global _graph
    if _graph is None:
        builder = StateGraph(AgentState)
        builder.add_node("rewrite", rewrite_node)
        builder.add_node("retrieve", retrieve_node)
        builder.add_node("generate", generate_node)
        builder.add_node("remember", remember_node)

        builder.add_edge(START, "rewrite")
        builder.add_edge("rewrite", "retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", "remember")
        builder.add_edge("remember", END)

        _graph = builder.compile(checkpointer=MemorySaver())
    return _graph


def run_agent(question: str, session_id: str = "default", k: Optional[int] = None) -> Dict:
    """Answer one message inside a conversation identified by session_id."""
    result = get_graph().invoke(
        {"question": question, "k": k},
        config={"configurable": {"thread_id": session_id}},
    )
    return {
        "question": question,
        "standalone_question": result["standalone_question"],
        "answer": result["answer"],
        "sources": result["sources"],
        "session_id": session_id,
    }


if __name__ == "__main__":
    # Demo: a follow-up question that only makes sense with memory
    demo = [
        "What is the penalty if Acme delivers late?",
        "And what is the maximum cap on that?",      # needs memory
        "Who needs to approve a $20,000 purchase?",  # topic switch
    ]
    for q in demo:
        print("=" * 70)
        out = run_agent(q, session_id="demo")
        print(f"User            : {q}")
        print(f"Understood as   : {out['standalone_question']}")
        print(f"Answer          : {out['answer']}")
