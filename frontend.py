import os
import streamlit as st
import backend as demo

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Memory + RAG Chatbot",
    page_icon="📌",
    layout="wide"
)

st.title("📌 Memory + RAG Chatbot")
st.caption("Powered by Gemini + LangChain + Chroma")

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:

    st.header("Controls")

    if st.button(
        "Clear Conversation",
        use_container_width=True
    ):

        st.session_state.chat_history = []

        if "memory" in st.session_state:
            del st.session_state.memory

        st.rerun()

    st.header("Upload PDFs")

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.divider()

    st.markdown(
        """
        **About this bot**

        - Uses PDF-based RAG
        - Uses Semantic Search
        - Uses Chroma Vector Database
        - Uses Conversation Memory
        - Model: Gemini 2.5 Flash
        """
    )

# ==========================================
# INITIALIZE CHATBOT
# ==========================================

os.makedirs(
    "uploaded_pdfs",
    exist_ok=True
)
if uploaded_files:

    if "retriever" not in st.session_state:

        with st.spinner(
            "Processing PDFs and creating knowledge base..."
        ):

            pdf_paths = []

            for uploaded_file in uploaded_files:

                file_path = os.path.join(
                    "uploaded_pdfs",
                    uploaded_file.name
                )

                with open(
                    file_path,
                    "wb"
                ) as f:

                    f.write(
                        uploaded_file.getbuffer()
                    )

                pdf_paths.append(
                    file_path
                )

            llm, memory, retriever = (
                demo.initialize_chatbot(
                    pdf_paths
                )
            )

            st.session_state.llm = llm
            st.session_state.memory = memory
            st.session_state.retriever = retriever

else:

    st.info(
        "Please upload one or more PDFs."
    )

    st.stop()

# ==========================================
# CHAT HISTORY
# ==========================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ==========================================
# DISPLAY OLD MESSAGES
# ==========================================

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):
        st.markdown(message["text"])

# ==========================================
# USER INPUT
# ==========================================

user_question = st.chat_input(
    "Ask a question about the PDF..."
)

if user_question:

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_question)

    st.session_state.chat_history.append(
        {
            "role": "user",
            "text": user_question
        }
    )

    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner("Searching PDF and thinking..."):

            response = demo.ask_question(
                question=user_question,
                retriever=st.session_state.retriever,
                memory=st.session_state.memory,
                llm=st.session_state.llm
            )

        st.markdown(response)

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "text": response
        }
    )