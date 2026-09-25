import streamlit as st


def render_document_panel(
    documents: list[dict],
    selected_document_id: str | None
):

    if selected_document_id is None:

        st.info(
            "Searching across all available documents."
        )
        return

    selected_document = None

    for document in documents:

        if document["document_id"] == selected_document_id:
            selected_document = document
            break

    if selected_document is None:

        st.warning(
            "Selected document information is unavailable."
        )
        return

    st.markdown("### Selected Document")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Document",
            selected_document["document_name"]
        )

    with col2:
        st.metric(
            "Pages",
            selected_document["pages"]
        )

    with col3:
        st.metric(
            "Chunks",
            selected_document["chunks"]
        )