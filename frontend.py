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

    if st.button("Clear Conversation",
                 use_container_width=True):

        llm, memory, retriever = demo.initialize_chatbot(
            "PDFs/NCHS Data Brief.pdf"
        )

        st.session_state.llm = llm
        st.session_state.memory = memory
        st.session_state.retriever = retriever

        st.session_state.chat_history = []

        st.rerun()

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

if "llm" not in st.session_state:

    with st.spinner("Loading PDF and creating vector database..."):

        llm, memory, retriever = demo.initialize_chatbot(
            "PDFs/NCHS Data Brief.pdf"
        )

        st.session_state.llm = llm
        st.session_state.memory = memory
        st.session_state.retriever = retriever

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