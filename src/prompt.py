"""
prompt.py
─────────
Defines the reusable RAG prompt template for government scheme queries.

Responsibilities
────────────────
• Provide a ChatPromptTemplate that instructs the LLM to:
    - Answer only from retrieved scheme context (no outside knowledge)
    - Never hallucinate or fabricate scheme details
    - Clearly state when information is unavailable
    - Cite the source scheme names and metadata
"""

from langchain_core.prompts import ChatPromptTemplate

# ── System instruction ────────────────────────────────────────────────────────
_SYSTEM_TEMPLATE = """\
You are an expert Government Scheme Assistant. Your mission is to help \
citizens understand government welfare schemes — their eligibility criteria, \
benefits, required documents, and application processes.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT RULES — YOU MUST FOLLOW THESE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Answer the user's query precisely and concisely using ONLY the provided \
context below. Do NOT use any prior knowledge or information outside the context.

2. If the context does not contain enough information to answer the question, \
politely state that you cannot find the answer in the scheme details. \
Say exactly:
   "I cannot find sufficient information about this in the available scheme \
details. Please try rephrasing your question or ask about a specific scheme name."

3. NEVER fabricate, guess, or hallucinate details about any government scheme. \
If you are unsure, admit it.

4. Always cite the scheme name and source(s) you used at the end of your answer in this format:
   📄 Source: <scheme_name> (File: <filename>, Page <page_number>)

5. Keep your answer clear, structured, and easy for a common citizen to understand. \
Use bullet points where helpful.

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
User Query: {query}

Answer strictly based on the context above and cite your sources at the end.
"""


def get_prompt_template() -> ChatPromptTemplate:
    """
    Build and return the RAG ChatPromptTemplate.

    The template expects three input variables:
        - context  : concatenated retrieved chunk texts
        - sources  : numbered source list (scheme name + metadata)
        - query    : the user's query

    Returns:
        A LangChain ChatPromptTemplate ready to be formatted.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_TEMPLATE),
            ("human", _HUMAN_TEMPLATE),
        ]
    )
