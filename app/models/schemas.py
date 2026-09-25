from pydantic import BaseModel


class QuestionRequest(BaseModel):
    question: str
    document_id: str | None = None


class SourceResponse(BaseModel):
    document_name: str
    page_number: int
    chunk_id: str
    score: float


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]