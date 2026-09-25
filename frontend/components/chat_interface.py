import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"


def render_chat_interface(
    selected_document_id: str | None
):

    st.subheader("Ask Your Documents")

    question = st.text_area(
        "Enter your question",
        placeholder="Example: What is the purpose of the employee handbook?",
        height=100
    )

    ask_button = st.button(
        "Ask Question",
        type="primary",
        use_container_width=True
    )

    if not ask_button:
        return

    if not question.strip():

        st.warning("Please enter a question.")
        return

    with st.spinner(
        "Searching documents and generating answer..."
    ):

        try:

            response = requests.post(
                f"{API_URL}/api/ask",
                json={
                    "question": question,
                    "document_id": selected_document_id
                },
                timeout=120
            )

            if response.status_code != 200:

                st.error(
                    response.json().get(
                        "detail",
                        "Failed to generate answer."
                    )
                )
                return

            result = response.json()

        except requests.exceptions.ConnectionError:

            st.error(
                "Cannot connect to FastAPI. "
                "Please make sure the backend is running."
            )
            return

        except Exception as error:

            st.error(
                f"Unexpected error: {error}"
            )
            return

    # -----------------------------
    # Answer
    # -----------------------------

    st.markdown("### Answer")

    st.markdown(
        f"""
        <div class="answer-card">
            {result["answer"]}
        </div>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------
    # Sources
    # -----------------------------

    sources = result.get("sources", [])

    if sources:

        st.markdown("### 📚 Sources")

        for index, source in enumerate(sources, start=1):

            with st.expander(
                f"Source {index} — "
                f"{source['document_name']} "
                f"(Page {source['page_number']})"
            ):

                st.write(
                    f"**Document:** {source['document_name']}"
                )

                st.write(
                    f"**Page:** {source['page_number']}"
                )

                st.write(
                    f"**Chunk:** {source['chunk_id']}"
                )

                st.write(
                    f"**Relevance Score:** "
                    f"{source['score']:.4f}"
                )

    else:

        st.info("No supporting sources were found.")