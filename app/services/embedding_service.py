from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def generate_embeddings(chunks: list[dict]) -> list[dict]:
    """
    Generate embeddings for document chunks.
    """

    if not chunks:
        raise ValueError("No chunks provided.")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()

    return chunks


if __name__ == "__main__":

    from app.services.extraction_service import extract_text_from_pdf
    from app.services.chunking_service import chunk_documents

    pdf_path = "data/documents/Employee_Handbook.pdf"

    pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_documents(pages)

    chunks_with_embeddings = generate_embeddings(chunks)

    print(f"\nChunks processed: {len(chunks_with_embeddings)}")
    print(f"Embedding dimension: {len(chunks_with_embeddings[0]['embedding'])}")

    print("\nFirst embedding:")
    print(chunks_with_embeddings[0]["embedding"][:10])