import streamlit as st
import requests

from frontend.components.sidebar import render_sidebar
from frontend.components.document_panel import render_document_panel
from frontend.components.chat_interface import render_chat_interface


API_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="RAG Document Intelligence",
    page_icon="📄",
    layout="wide"
)


# -----------------------------
# Custom Styling
# -----------------------------

st.markdown(
    """
    <style>

        .main-title {
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 4px;
        }

        .subtitle {
            color: #6b7280;
            font-size: 16px;
            margin-bottom: 25px;
        }

        .answer-card {
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #e5e7eb;
            background-color: #fafafa;
            line-height: 1.7;
            color: #111827;
        }

        .source-card {
            padding: 12px 15px;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            margin-bottom: 10px;
        }

    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Header
# -----------------------------

st.markdown(
    '<div class="main-title">'
    '📄 RAG Document Intelligence'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload documents, search their knowledge, and ask questions using AI.'
    '</div>',
    unsafe_allow_html=True
)


# -----------------------------
# Sidebar
# -----------------------------

selected_document_id = render_sidebar()


# -----------------------------
# Get Documents
# -----------------------------

try:

    response = requests.get(
        f"{API_URL}/api/documents",
        timeout=10
    )

    if response.status_code == 200:

        documents = response.json().get(
            "documents",
            []
        )

    else:

        documents = []

except Exception:

    documents = []


# -----------------------------
# Document Information
# -----------------------------

render_document_panel(
    documents=documents,
    selected_document_id=selected_document_id
)


st.divider()


# -----------------------------
# Chat Interface
# -----------------------------

render_chat_interface(
    selected_document_id=selected_document_id
)