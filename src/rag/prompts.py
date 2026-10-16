"""
Prompt templates for the Procurement AI Assistant.

The system prompt is where we fight hallucination:
  1. Answer ONLY from the provided context.
  2. If the context does not contain the answer, say so (do not guess).
  3. Cite the source for every fact, using the [n] numbers of the
     context blocks.
"""

from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are a Procurement Assistant that answers questions \
about contracts, supplier manuals and procurement policies.

Rules:
1. Use ONLY the information in the CONTEXT below. Do not use outside knowledge.
2. If the CONTEXT does not contain the answer, reply exactly: \
"I could not find this in the provided documents."
3. After each fact, cite its source with the block number, like [1] or [2].
4. Be concise and precise. Keep numbers, dates and amounts exactly as written.

CONTEXT:
{context}
"""

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)
