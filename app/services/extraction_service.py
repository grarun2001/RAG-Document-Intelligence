import fitz
import pytesseract

from pathlib import Path
from PIL import Image
from io import BytesIO


def extract_text_from_pdf(file_path: str) -> list[dict]:

    pdf_path = Path(file_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    pages = []

    with fitz.open(pdf_path) as document:

        if len(document) == 0:
            raise ValueError("PDF contains no pages.")

        for page_number, page in enumerate(document, start=1):

            text = page.get_text("text").strip()

            # Normal PDF text extraction
            if text:
                pages.append({
                    "page_number": page_number,
                    "text": text,
                    "extraction_method": "text"
                })

                continue

            # OCR fallback
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

            image_bytes = pix.tobytes("png")

            image = Image.open(BytesIO(image_bytes))

            ocr_text = pytesseract.image_to_string(image).strip()

            if ocr_text:
                pages.append({
                    "page_number": page_number,
                    "text": ocr_text,
                    "extraction_method": "ocr"
                })

    if not pages:
        raise ValueError(
            "No searchable text could be extracted from this PDF."
        )

    return pages


if __name__ == "__main__":

    pdf_path = "data/documents/Employee_Handbook.pdf"

    pages = extract_text_from_pdf(pdf_path)

    print(f"Pages extracted: {len(pages)}")

    for page in pages[:2]:

        print("\n" + "=" * 60)

        print(
            f"Page {page['page_number']} "
            f"({page['extraction_method']})"
        )

        print("=" * 60)

        print(page["text"][:1000])