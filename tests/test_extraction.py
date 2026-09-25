from app.services.extraction_service import extract_text_from_pdf


def test_pdf_extraction():

    pages = extract_text_from_pdf(
        "data/documents/Employee_Handbook.pdf"
    )

    assert len(pages) > 0
    assert "text" in pages[0]
    assert "page_number" in pages[0]