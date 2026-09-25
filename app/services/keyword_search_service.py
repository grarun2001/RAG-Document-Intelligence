from sentence_transformers import SentenceTransformer
from pymilvus import Collection

from rank_bm25 import BM25Okapi

from app.services.vector_service import (
    connect_to_milvus,
    COLLECTION_NAME
)


EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# --------------------------------------------------
# Load embedding model
# --------------------------------------------------

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


# --------------------------------------------------
# Simple tokenizer
# --------------------------------------------------

def tokenize(text: str) -> list[str]:
    """
    Convert text into lowercase tokens.
    """

    return text.lower().split()


# --------------------------------------------------
# Fetch all chunks from Milvus
# --------------------------------------------------

def get_all_chunks(
    document_id: str | None = None
) -> list[dict]:
    """
    Fetch document chunks from Milvus
    for keyword searching.
    """

    connect_to_milvus()

    collection = Collection(COLLECTION_NAME)

    collection.load()

    # ----------------------------------------------
    # Build document filter
    # ----------------------------------------------

    expression = None

    if document_id:
        expression = (
            f'document_id == "{document_id}"'
        )

    # ----------------------------------------------
    # Query stored chunks
    # ----------------------------------------------

    if expression:

        results = collection.query(
            expr=expression,
            output_fields=[
                "document_id",
                "document_name",
                "page_number",
                "chunk_id",
                "text"
            ],
            limit=16384
        )

    else:

        results = collection.query(
            expr='document_id != ""',
            output_fields=[
                "document_id",
                "document_name",
                "page_number",
                "chunk_id",
                "text"
            ],
            limit=16384
        )

    chunks = []

    for item in results:

        chunks.append({
            "document_id": item.get("document_id"),
            "document_name": item.get("document_name"),
            "page_number": item.get("page_number"),
            "chunk_id": item.get("chunk_id"),
            "text": item.get("text")
        })

    return chunks


# --------------------------------------------------
# Keyword Search using BM25
# --------------------------------------------------

def keyword_search(
    question: str,
    document_id: str | None = None,
    top_k: int = 10
) -> list[dict]:
    """
    Perform keyword-based search using BM25.
    """

    if not question.strip():

        raise ValueError(
            "Question cannot be empty."
        )

    chunks = get_all_chunks(
        document_id=document_id
    )

    if not chunks:
        return []

    # ----------------------------------------------
    # Prepare corpus
    # ----------------------------------------------

    corpus = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    # ----------------------------------------------
    # Create BM25 index
    # ----------------------------------------------

    bm25 = BM25Okapi(corpus)

    query_tokens = tokenize(question)

    scores = bm25.get_scores(
        query_tokens
    )

    # ----------------------------------------------
    # Attach keyword score
    # ----------------------------------------------

    scored_chunks = []

    for chunk, score in zip(
        chunks,
        scores
    ):

        chunk_copy = chunk.copy()

        chunk_copy["keyword_score"] = float(
            score
        )

        scored_chunks.append(
            chunk_copy
        )

    # ----------------------------------------------
    # Sort highest score first
    # ----------------------------------------------

    scored_chunks.sort(
        key=lambda x: x["keyword_score"],
        reverse=True
    )

    return scored_chunks[:top_k]


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    question = (
        "What types of leave are available "
        "to employees?"
    )

    results = keyword_search(
        question=question,
        top_k=5
    )

    print(
        f"\nQuestion: {question}"
    )

    print(
        f"Results found: {len(results)}"
    )

    for index, result in enumerate(
        results,
        start=1
    ):

        print("\n" + "=" * 60)

        print(
            f"Result       : {index}"
        )

        print(
            f"Document     : "
            f"{result['document_name']}"
        )

        print(
            f"Page         : "
            f"{result['page_number']}"
        )

        print(
            f"Chunk ID      : "
            f"{result['chunk_id']}"
        )

        print(
            f"Keyword Score: "
            f"{result['keyword_score']:.4f}"
        )

        print("=" * 60)

        print(
            result["text"][:500]
        )