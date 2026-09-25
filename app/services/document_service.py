from pathlib import Path
import json
import uuid

from app.services.extraction_service import extract_text_from_pdf
from app.services.chunking_service import chunk_documents
from app.services.embedding_service import generate_embeddings
from app.services.vector_service import insert_chunks


DOCUMENTS_DIR = Path("data/documents")
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

REGISTRY_FILE = DOCUMENTS_DIR / "documents.json"


def _load_registry() -> list[dict]:

    if not REGISTRY_FILE.exists():
        return []

    with open(REGISTRY_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def _save_registry(documents: list[dict]):

    with open(REGISTRY_FILE, "w", encoding="utf-8") as file:
        json.dump(documents, file, indent=4)


def process_document(file_path: str, document_name: str) -> dict:

    documents = _load_registry()

    # Duplicate filename check
    for document in documents:

        if document["document_name"].lower() == document_name.lower():

            return {
                **document,
                "status": "already_exists"
            }

    document_id = str(uuid.uuid4())

    pages = extract_text_from_pdf(file_path)

    chunks = chunk_documents(pages)

    chunks = generate_embeddings(chunks)

    insert_chunks(
        chunks=chunks,
        document_id=document_id,
        document_name=document_name
    )

    document_info = {
        "document_id": document_id,
        "document_name": document_name,
        "file_path": str(file_path),
        "pages": len(pages),
        "chunks": len(chunks)
    }

    documents.append(document_info)

    _save_registry(documents)

    return {
        **document_info,
        "status": "processed"
    }


def get_documents() -> list[dict]:

    return _load_registry()