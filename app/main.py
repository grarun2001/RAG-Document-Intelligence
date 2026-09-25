from fastapi import FastAPI

from app.api.routes import router


app = FastAPI(
    title="RAG Document Intelligence API",
    description="Professional PDF-based RAG Question Answering API",
    version="1.0.0"
)


app.include_router(router)


@app.get("/")
def root():
    return {
        "message": "RAG Document Intelligence API is running."
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }