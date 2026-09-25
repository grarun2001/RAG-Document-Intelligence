from sentence_transformers import CrossEncoder


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

reranker = CrossEncoder(RERANKER_MODEL)


def rerank_chunks(
    question: str,
    chunks: list[dict],
    top_k: int = 3
) -> list[dict]:

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not chunks:
        return []

    pairs = [
        (question, chunk["text"])
        for chunk in chunks
    ]

    scores = reranker.predict(pairs)

    reranked_chunks = []

    for chunk, score in zip(chunks, scores):
        chunk_copy = chunk.copy()
        chunk_copy["rerank_score"] = float(score)
        reranked_chunks.append(chunk_copy)

    reranked_chunks.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return reranked_chunks[:top_k]


if __name__ == "__main__":

    from app.services.retrieval_service import retrieve_chunks

    question = "What is the purpose of the employee handbook?"

    print("Retrieving relevant chunks...")

    retrieved_chunks = retrieve_chunks(
        question=question,
        top_k=5
    )

    print(f"Retrieved chunks: {len(retrieved_chunks)}")

    print("\nReranking chunks...")

    reranked_chunks = rerank_chunks(
        question=question,
        chunks=retrieved_chunks,
        top_k=3
    )

    print(f"Final chunks: {len(reranked_chunks)}")

    for index, chunk in enumerate(reranked_chunks, start=1):

        print("\n" + "=" * 60)
        print(f"Rank       : {index}")
        print(f"Document   : {chunk['document_name']}")
        print(f"Page       : {chunk['page_number']}")
        print(f"Chunk ID   : {chunk['chunk_id']}")
        print(f"Rerank score: {chunk['rerank_score']:.4f}")
        print("=" * 60)

        print(chunk["text"][:500])