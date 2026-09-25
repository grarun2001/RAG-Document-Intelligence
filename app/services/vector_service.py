from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)


COLLECTION_NAME = "rag_documents"
EMBEDDING_DIMENSION = 384


def connect_to_milvus():
    connections.connect(
        alias="default",
        host="localhost",
        port="19530"
    )


def create_collection():
    connect_to_milvus()

    if utility.has_collection(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists.")
        return Collection(COLLECTION_NAME)

    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True
        ),
        FieldSchema(
            name="document_id",
            dtype=DataType.VARCHAR,
            max_length=200
        ),
        FieldSchema(
            name="document_name",
            dtype=DataType.VARCHAR,
            max_length=500
        ),
        FieldSchema(
            name="page_number",
            dtype=DataType.INT64
        ),
        FieldSchema(
            name="chunk_id",
            dtype=DataType.VARCHAR,
            max_length=200
        ),
        FieldSchema(
            name="text",
            dtype=DataType.VARCHAR,
            max_length=10000
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=EMBEDDING_DIMENSION
        )
    ]

    schema = CollectionSchema(
        fields=fields,
        description="RAG document chunks and embeddings"
    )

    collection = Collection(
        name=COLLECTION_NAME,
        schema=schema
    )

    print(f"Collection '{COLLECTION_NAME}' created successfully!")

    return collection


def insert_chunks(chunks, document_id, document_name):
    collection = create_collection()

    data = [
        [document_id] * len(chunks),
        [document_name] * len(chunks),
        [chunk["page_number"] for chunk in chunks],
        [chunk["chunk_id"] for chunk in chunks],
        [chunk["text"] for chunk in chunks],
        [chunk["embedding"] for chunk in chunks]
    ]

    collection.insert(data)

    collection.flush()

    print(f"{len(chunks)} chunks inserted successfully.")


if __name__ == "__main__":

    from app.services.extraction_service import extract_text_from_pdf
    from app.services.chunking_service import chunk_documents
    from app.services.embedding_service import generate_embeddings

    pdf_path = "data/documents/Employee_Handbook.pdf"

    print("1. Extracting PDF...")
    pages = extract_text_from_pdf(pdf_path)

    print(f"   Pages extracted: {len(pages)}")

    print("\n2. Creating chunks...")
    chunks = chunk_documents(pages)

    print(f"   Chunks created: {len(chunks)}")

    print("\n3. Generating embeddings...")
    chunks = generate_embeddings(chunks)

    print(f"   Embeddings generated: {len(chunks)}")

    print("\n4. Inserting into Milvus...")

    insert_chunks(
        chunks=chunks,
        document_id="employee_handbook_001",
        document_name="Employee_Handbook.pdf"
    )

    print("\nDocument ingestion completed successfully!")