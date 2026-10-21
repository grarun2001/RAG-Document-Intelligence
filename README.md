# RAG Document Intelligence

An AI-powered document question-answering system that uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from uploaded documents and generate context-aware answers.

The application combines document processing, OCR-based text extraction, semantic search, and large language models to help users query their documents using natural language.

## Overview

Searching through lengthy documents manually can be time-consuming. This project simplifies document exploration by allowing users to upload documents, ask questions, and retrieve relevant information from the indexed content.

Instead of relying only on an LLM's pretrained knowledge, the RAG pipeline retrieves relevant document context before generating an answer.

## Key Features

- **Document Ingestion:** Process uploaded documents and prepare their content for retrieval.
- **PDF Text Extraction:** Extract text from PDFs using PyMuPDF.
- **OCR Support:** Use Tesseract OCR as a fallback for scanned or image-based documents.
- **Text Chunking:** Split extracted content into smaller chunks for effective retrieval.
- **Semantic Search:** Generate embeddings and retrieve relevant document passages.
- **Vector Database:** Use Milvus to store and search document embeddings.
- **LLM-Based Question Answering:** Use Google Gemini to generate answers grounded in retrieved context.
- **Hybrid Retrieval:** Combine semantic retrieval with BM25 keyword search to improve relevant context discovery.
- **Source References:** Support source-based answers so users can verify retrieved information.
- **API Backend:** Use FastAPI to expose document-processing and question-answering functionality.
- **Interactive Interface:** Provide a Streamlit-based interface for document interaction.
- **Testing:** Include tests to validate supported application components.

## System Architecture

```text
             User
              |
              v
       Streamlit Interface
              |
              v
          FastAPI
              |
       Document Upload
              |
              v
      Document Processing
              |
       +------+------+
       |             |
       v             v
   PDF Text       OCR Fallback
   Extraction     (Tesseract)
       |             |
       +------+------+
              |
              v
       Text Cleaning
       and Chunking
              |
              v
        Embeddings
              |
              v
        Milvus Store
              |
              v
        User Question
              |
              v
      Hybrid Retrieval
   Semantic Search + BM25
              |
              v
       Relevant Context
              |
              v
        Google Gemini
              |
              v
      Answer with Sources
```

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| LLM | Google Gemini |
| Backend API | FastAPI |
| User Interface | Streamlit |
| Document Processing | PyMuPDF |
| OCR | Tesseract OCR |
| Text Splitting | LangChain Text Splitters |
| Embeddings | Sentence Transformers |
| Vector Database | Milvus |
| Keyword Retrieval | BM25 |
| Data Validation | Pydantic |
| Testing | Pytest |

## Project Structure

```text
RAG-Document-Intelligence/
├── app/
│   └── Application modules
├── data/
│   └── documents/
├── docker/
├── frontend/
├── tests/
├── .env.example
├── .gitignore
├── requirements.txt
├── run.py
└── README.md
```

## How It Works

### 1. Document Ingestion
Users upload documents through the application interface. The backend processes the uploaded content for indexing.

### 2. Text Extraction
Text is extracted from supported PDFs. OCR can be used when scanned documents do not contain directly extractable text.

### 3. Chunking and Embeddings
The extracted text is divided into smaller chunks. An embedding model converts these chunks into vector representations.

### 4. Indexing and Retrieval
Document embeddings are stored in Milvus. When a user asks a question, the retrieval pipeline searches for relevant passages using semantic and keyword-based retrieval.

### 5. Answer Generation
The retrieved passages are supplied as context to Google Gemini, which generates a response based on the available evidence.

### 6. Source Verification
Retrieved document context and source references help users inspect the information supporting the generated answer.

## Prerequisites

- Python 3.10 or a compatible version for the installed dependencies
- Google Gemini API key
- Milvus vector database
- Tesseract OCR installed if OCR functionality is required
- Git

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/grarun2001/RAG-Document-Intelligence.git
cd RAG-Document-Intelligence
```

### 2. Create a Virtual Environment

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file based on `.env.example`.

```env
GEMINI_API_KEY=your_gemini_api_key_here
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

Replace the placeholder with your own Gemini API key. Do not commit `.env` or API keys to GitHub.

### 5. Start the Required Services

Ensure that your Milvus instance is running and reachable using the configured host and port.

### 6. Run the Application

Start the FastAPI backend and Streamlit interface using the actual entry-point modules configured in the project.

Refer to the application files for the correct startup commands and available endpoints.

## Testing

Run the test suite from the project root:

```bash
python -m pytest -v
```

## Use Cases

- Searching employee handbooks and policy documents
- Extracting information from PDF reports
- Asking questions about technical documentation
- Exploring scanned documents using OCR
- Retrieving relevant information from document collections

## Future Improvements

- Expand support for additional document formats.
- Improve OCR accuracy for complex layouts and handwritten content.
- Add retrieval evaluation metrics to measure search relevance.
- Improve document-level filtering and source attribution.
- Add caching and asynchronous processing where appropriate.

## Author

**Arun GR**

GitHub: [grarun2001](https://github.com/grarun2001)

---

*This project demonstrates the integration of document processing, information retrieval, vector search, and LLM-based question answering in a RAG application.*
