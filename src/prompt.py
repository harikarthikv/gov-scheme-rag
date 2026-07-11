"""
prompt.py
─────────
Defines the reusable RAG prompt template.

Responsibilities
────────────────
• Provide a ChatPromptTemplate that instructs the LLM to:
    - Answer only from retrieved context (no outside knowledge)
    - Never hallucinate or fabricate scheme details
    - Clearly state when information is unavailable
    - Cite the source documents (filename + page number)
"""

from langchain_core.prompts import ChatPromptTemplate

# ── System instruction ────────────────────────────────────────────────────────
_SYSTEM_TEMPLATE = """\
You are an expert assistant specialising in Indian Government Schemes.
Your mission is to help Indian citizens clearly understand government welfare \
schemes, their eligibility criteria, benefits, and application processes.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT RULES — YOU MUST FOLLOW THESE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. ONLY use information from the CONTEXT provided below.
   Do NOT use any prior knowledge or information outside the context.

2. If the context does not contain enough information to answer the question,
   respond with:
   "I don't have sufficient information in the provided documents to answer \
this question."

3. NEVER fabricate, guess, or hallucinate details about any government scheme.

4. Always cite the source(s) you used at the end of your answer in this format:
   📄 Source: <filename>, Page <page_number>

5. Keep your answer clear, structured, and easy for a common citizen to understand.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTEXT (retrieved from official scheme documents):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RETRIEVED SOURCES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{sources}
"""

# ── Human turn ────────────────────────────────────────────────────────────────
_HUMAN_TEMPLATE = """\
Question: {question}

Answer strictly based on the context above and cite your sources at the end.
"""


def get_prompt_template() -> ChatPromptTemplate:
    """
    Build and return the RAG ChatPromptTemplate.

    The template expects three input variables:
        - context  : concatenated retrieved chunk texts
        - sources  : numbered source list (filename + page)
        - question : the user's query

    Returns:
        A LangChain ChatPromptTemplate ready to be formatted.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_TEMPLATE),
            ("human", _HUMAN_TEMPLATE),
        ]
    )
