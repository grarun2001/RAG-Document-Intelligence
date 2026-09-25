from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File

from app.models.schemas import (
    QuestionRequest,
    AnswerResponse
)

from app.services.rag_service import generate_answer
from app.services.document_service import (
    process_document,
    get_documents
)


router = APIRouter(
    prefix="/api",
    tags=["RAG"]
)


DOCUMENTS_DIR = Path("data/documents")
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):

    try:

        if not request.question.strip():

            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty."
            )

        result = generate_answer(
            question=request.question,
            document_id=request.document_id
        )

        return result

    except HTTPException:

        raise

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate answer: {str(error)}"
        )


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file selected."
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    file_path = DOCUMENTS_DIR / file.filename

    try:

        file_content = await file.read()

        with open(file_path, "wb") as buffer:
            buffer.write(file_content)

        result = process_document(
            file_path=str(file_path),
            document_name=file.filename
        )

        return result

    except Exception as error:

        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {str(error)}"
        )
        
@router.get("/documents")
def list_documents():

    try:
        return {
            "documents": get_documents()
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch documents: {str(error)}"
        )