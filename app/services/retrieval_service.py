from sentence_transformers import SentenceTransformer
from pymilvus import Collection

from app.services.vector_service import (
    connect_to_milvus,
    COLLECTION_NAME
)


EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

model = SentenceTransformer(EMBEDDING_MODEL)


def retrieve_chunks(
    question: str,
    document_id: str | None = None,
    top_k: int = TOP_K
) -> list[dict]:
    """
    Retrieve the most relevant chunks from Milvus.
    """

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    connect_to_milvus()

    collection = Collection(COLLECTION_NAME)

    # Create vector index if it does not exist
    if not collection.indexes:
        collection.create_index(
            field_name="embedding",
            index_params={
                "index_type": "AUTOINDEX",
                "metric_type": "COSINE",
                "params": {}
            }
        )

    collection.load()

    # Convert question into embedding
    question_embedding = model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).tolist()

    # Filter by document if a specific document is selected
    search_params = {
        "metric_type": "COSINE",
        "params": {}
    }

    filter_expression = None

    if document_id:
        filter_expression = f'document_id == "{document_id}"'

    results = collection.search(
        data=[question_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        expr=filter_expression,
        output_fields=[
            "document_id",
            "document_name",
            "page_number",
            "chunk_id",
            "text"
        ]
    )

    retrieved_chunks = []

    for hits in results:
        for hit in hits:
            retrieved_chunks.append({
                "document_id": hit.entity.get("document_id"),
                "document_name": hit.entity.get("document_name"),
                "page_number": hit.entity.get("page_number"),
                "chunk_id": hit.entity.get("chunk_id"),
                "text": hit.entity.get("text"),
                "score": float(hit.distance)
            })

    return retrieved_chunks


if __name__ == "__main__":

    question = "What is the purpose of the employee handbook?"

    results = retrieve_chunks(question)

    print(f"\nQuestion: {question}")
    print(f"Results found: {len(results)}")

    for index, result in enumerate(results, start=1):

        print("\n" + "=" * 60)
        print(f"Result {index}")
        print("=" * 60)

        print(f"Document : {result['document_name']}")
        print(f"Page     : {result['page_number']}")
        print(f"Chunk ID : {result['chunk_id']}")
        print(f"Score    : {result['score']:.4f}")

        print("\nText:")
        print(result["text"][:500])