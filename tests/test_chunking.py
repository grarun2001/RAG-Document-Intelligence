from app.services.extraction_service import extract_text_from_pdf
from app.services.chunking_service import chunk_documents


def test_chunking():

    pages = extract_text_from_pdf(
        "data/documents/Employee_Handbook.pdf"
    )

    chunks = chunk_documents(pages)

    assert len(chunks) > 0
    assert "text" in chunks[0]
    assert "page_number" in chunks[0]