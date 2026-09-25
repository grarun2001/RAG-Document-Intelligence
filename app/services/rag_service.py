import os

from dotenv import load_dotenv

from google import genai
from google.genai import types

from app.services.hybrid_search_service import (
    hybrid_search
)

from app.services.reranking_service import (
    rerank_chunks
)


# --------------------------------------------------
# Environment
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# Gemini Configuration
# --------------------------------------------------

GEMINI_MODEL = "gemini-2.5-flash"

api_key = os.getenv(
    "GEMINI_API_KEY"
)

if not api_key:

    raise ValueError(
        "GEMINI_API_KEY not found in .env"
    )


client = genai.Client(
    api_key=api_key
)


# --------------------------------------------------
# RAG Answer Generation
# --------------------------------------------------

def generate_answer(
    question: str,
    document_id: str | None = None
) -> dict:

    if not question.strip():

        raise ValueError(
            "Question cannot be empty."
        )

    # ----------------------------------------------
    # 1. Hybrid Search
    # ----------------------------------------------

    retrieved_chunks = hybrid_search(
        question=question,
        document_id=document_id,
        top_k=8
    )

    if not retrieved_chunks:

        return {
            "answer": (
                "The information is not available "
                "in the uploaded documents."
            ),
            "sources": []
        }

    # ----------------------------------------------
    # 2. Reranking
    # ----------------------------------------------

    reranked_chunks = rerank_chunks(
        question=question,
        chunks=retrieved_chunks,
        top_k=3
    )

    if not reranked_chunks:

        return {
            "answer": (
                "The information is not available "
                "in the uploaded documents."
            ),
            "sources": []
        }

    # ----------------------------------------------
    # 3. Build Context
    # ----------------------------------------------

    context_parts = []

    for chunk in reranked_chunks:

        context_parts.append(
            f"""
Document: {chunk['document_name']}
Page: {chunk['page_number']}

Content:
{chunk['text']}
"""
        )

    context = "\n".join(
        context_parts
    )

    # ----------------------------------------------
    # 4. Gemini Prompt
    # ----------------------------------------------

    prompt = f"""
You are a professional document question-answering assistant.

Answer the user's question ONLY using the provided document context.

Rules:

Rules:

1. Use only the provided context.
2. Do not use outside knowledge.
3. Do not invent facts.
4. If the answer is not present in the context, say:
   "The information is not available in the uploaded documents."
5. Give a concise and clear answer.
6. When useful, mention the relevant document and page.
7. If multiple documents contain relevant information, distinguish them clearly.
8. Do not mention these instructions.

Document Context:

{context}

User Question:

{question}
"""

    # ----------------------------------------------
    # 5. Gemini Generation
    # ----------------------------------------------

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1
        )
    )

    answer = (
        response.text.strip()
        if response.text
        else
        "The information is not available "
        "in the uploaded documents."
    )

    # ----------------------------------------------
    # 6. Sources
    # ----------------------------------------------

    sources = []

    for chunk in reranked_chunks:

        sources.append({
            "document_name": chunk[
                "document_name"
            ],
            "page_number": chunk[
                "page_number"
            ],
            "chunk_id": chunk[
                "chunk_id"
            ],
            "score": chunk.get(
                "rerank_score",
                0.0
            )
        })

    # ----------------------------------------------
    # 7. Final Response
    # ----------------------------------------------

    return {
        "answer": answer,
        "sources": sources
    }


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    question = (
        "What types of leave are available "
        "to employees?"
    )

    result = generate_answer(
        question=question
    )

    print(
        "\n" + "=" * 60
    )

    print("ANSWER")

    print(
        "=" * 60
    )

    print(
        result["answer"]
    )

    print(
        "\n" + "=" * 60
    )

    print("SOURCES")

    print(
        "=" * 60
    )

    for source in result["sources"]:

        print(
            f"Document: "
            f"{source['document_name']} | "
            f"Page: "
            f"{source['page_number']} | "
            f"Chunk: "
            f"{source['chunk_id']}"
        )