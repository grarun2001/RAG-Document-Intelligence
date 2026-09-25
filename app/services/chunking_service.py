from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(pages: list[dict]) -> list[dict]:
    """
    Split extracted PDF pages into smaller chunks.

    Each chunk keeps:
    - page_number
    - chunk_id
    - text
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []

    for page in pages:

        page_number = page["page_number"]
        text = page["text"]

        page_chunks = splitter.split_text(text)

        for index, chunk_text in enumerate(page_chunks):

            chunks.append({
                "page_number": page_number,
                "chunk_id": f"page_{page_number}_chunk_{index}",
                "text": chunk_text.strip()
            })

    if not chunks:
        raise ValueError("No chunks were created from the document.")

    return chunks


if __name__ == "__main__":

    from app.services.extraction_service import extract_text_from_pdf

    pdf_path = "data/documents/Employee_Handbook.pdf"

    pages = extract_text_from_pdf(pdf_path)

    chunks = chunk_documents(pages)

    print(f"Pages extracted : {len(pages)}")
    print(f"Chunks created  : {len(chunks)}")

    for chunk in chunks[:3]:
        print("\n" + "=" * 60)
        print(f"Page     : {chunk['page_number']}")
        print(f"Chunk ID : {chunk['chunk_id']}")
        print("=" * 60)
        print(chunk["text"][:500])