"""Prompt used to keep answers tied to retrieved PDF content."""

from langchain_core.prompts import ChatPromptTemplate

_SYSTEM_TEMPLATE = """You answer questions about government schemes using only the context below.

If the context is insufficient, reply exactly:
"I cannot find sufficient information about this in the available scheme details. Please try rephrasing your question or ask about a specific scheme name."

Do not add facts that are not in the context. Write clearly and cite the relevant scheme and page at the end of the answer.

Context:
{context}

Retrieved sources:
{sources}
"""


def get_prompt_template() -> ChatPromptTemplate:
    """Return the chat prompt used for each question."""
    return ChatPromptTemplate.from_messages(
        [("system", _SYSTEM_TEMPLATE), ("human", "Question: {query}")]
    )
