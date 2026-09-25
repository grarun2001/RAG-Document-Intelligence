import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"


def render_sidebar():

    with st.sidebar:

        st.header("Document Management")

        uploaded_file = st.file_uploader(
            "Upload PDF",
            type=["pdf"]
        )

        if uploaded_file is not None:

            if st.button(
                "Process Document",
                use_container_width=True
            ):

                with st.spinner("Processing document..."):

                    try:

                        response = requests.post(
                            f"{API_URL}/api/upload",
                            files={
                                "file": (
                                    uploaded_file.name,
                                    uploaded_file.getvalue(),
                                    "application/pdf"
                                )
                            },
                            timeout=300
                        )

                        if response.status_code == 200:

                            result = response.json()

                            if result["status"] == "already_exists":

                                st.warning(
                                    f"{result['document_name']} is already in the knowledge base."
                                )

                            else:

                                st.success(
                                    "Document processed successfully."
                                )

                            st.write(
                                f"Pages: {result['pages']}"
                            )

                            st.write(
                                f"Chunks: {result['chunks']}"
                            )

                            st.rerun()

                        else:

                            try:
                                detail = response.json().get(
                                    "detail",
                                    "Document processing failed."
                                )
                            except Exception:
                                detail = "Document processing failed."

                            st.error(detail)

                    except Exception as error:

                        st.error(
                            f"Connection error: {error}"
                        )

        # Clear Chat button
        if st.button(
            "Clear Chat",
            use_container_width=True
        ):

            st.session_state.chat_history = []

            st.rerun()

        st.divider()

        # Knowledge Base
        st.subheader("🔎 Search Scope")

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

        document_options = ["All Documents"]

        document_map = {
            "All Documents": None
        }

        for document in documents:

            name = document["document_name"]

            document_options.append(name)

            document_map[name] = document["document_id"]

        selected_document = st.selectbox(
            "Choose knowledge source",
            document_options
        )

        selected_document_id = document_map[
            selected_document
        ]

        return selected_document_id