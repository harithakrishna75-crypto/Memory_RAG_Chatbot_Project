# ==========================================
# IMPORT REQUIRED LIBRARIES
# ==========================================

import os
from dotenv import load_dotenv

# Gemini LLM
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

# Memory
from langchain_classic.memory import ConversationSummaryBufferMemory

# PDF Loader
from langchain_community.document_loaders import PyPDFLoader

# Splits large documents into smaller chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector Database
from langchain_chroma import Chroma


# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

# Reads GOOGLE_API_KEY from .env file
load_dotenv()


# ==========================================
# CREATE GEMINI MODEL
# ==========================================

def create_llm():
    """
    Creates and returns Gemini LLM.

    This model will:
    - Read user questions
    - Read retrieved PDF content
    - Read conversation history
    - Generate final answer
    """

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found in .env file"
        )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=api_key,
        max_output_tokens=1024
    )

    return llm


# ==========================================
# CREATE MEMORY
# ==========================================

def create_memory():
    """
    Memory stores previous conversations.

    ConversationSummaryBufferMemory:
    - Keeps recent messages exactly
    - Summarizes old messages automatically
    - Prevents memory from becoming too large

    Example:

    User: What is mean?
    Bot: Mean is average.

    User: Explain it again.

    Memory remembers the earlier discussion.
    """

    llm = create_llm()

    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=1000,
        return_messages=False,
        memory_key="history",
        input_key="input"
    )

    return memory


# ==========================================
# LOAD PDFS AND CREATE RETRIEVER
# ==========================================

def create_retriever(pdf_paths):
    """
    Creates the RAG retrieval system.

    Steps:
    1. Load multiple PDFs
    2. Add metadata
    3. Split into chunks
    4. Create embeddings
    5. Store in Chroma
    6. Return retriever
    """

    documents = []

    # --------------------------------------
    # LOAD ALL PDFs
    # --------------------------------------

    for pdf_path in pdf_paths:

        loader = PyPDFLoader(pdf_path)

        docs = loader.load()

        # Add filename metadata
        for doc in docs:

            doc.metadata["source"] = os.path.basename(
                pdf_path
            )

            # Page already exists in metadata
            # but ensure it is available

            doc.metadata["page"] = (
                doc.metadata.get("page", 0) + 1
            )

        documents.extend(docs)

    # --------------------------------------
    # SPLIT DOCUMENTS
    # --------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    splits = splitter.split_documents(
        documents
    )

    # --------------------------------------
    # CREATE EMBEDDINGS
    # --------------------------------------

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    # --------------------------------------
    # CREATE VECTOR DATABASE
    # --------------------------------------

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )

    # --------------------------------------
    # CREATE RETRIEVER
    # --------------------------------------

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    return retriever

# ==========================================
# MAIN PROMPT
# ==========================================

PROMPT_TEMPLATE = """
You are a helpful AI assistant.

Use the retrieved PDF context whenever relevant.

Conversation History:
{history}

Retrieved PDF Context:
{context}

User Question:
{question}

Instructions:
- Answer using the PDF when possible.
- Use conversation history when useful.
- If answer is not found in PDF, say so so clearly and then provide a general answer
based on your knowledge.
- Keep answers clear and concise.

Answer:
"""


# ==========================================
# ASK QUESTION
# ==========================================

def ask_question(
    question,
    retriever,
    memory,
    llm
):
    """
    Main function that powers the chatbot.

    Flow:

    User Question
          ↓
    Retrieve PDF Chunks
          ↓
    Read Memory
          ↓
    Build Prompt
          ↓
    Gemini
          ↓
    Save Response To Memory
          ↓
    Return Answer
    """

    # --------------------------------------
    # STEP 1: RETRIEVE PDF CONTENT
    # --------------------------------------

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    # --------------------------------------
    # STEP 2: LOAD MEMORY
    # --------------------------------------

    history = memory.load_memory_variables(
        {}
    )["history"]

    # --------------------------------------
    # STEP 3: BUILD PROMPT
    # --------------------------------------

    prompt = PROMPT_TEMPLATE.format(
        history=history,
        context=context,
        question=question
    )

    # --------------------------------------
    # STEP 4: SEND TO GEMINI
    # --------------------------------------

    response = llm.invoke(prompt)

    answer = response.content

    # --------------------------------------
    # STEP 5: SAVE TO MEMORY
    # --------------------------------------

    memory.save_context(
        {"input": question},
        {"output": answer}
    )

    # --------------------------------------
    # STEP 6: RETURN ANSWER
    # --------------------------------------

    return answer


# ==========================================
# INITIALIZE EVERYTHING
# ==========================================

def initialize_chatbot(pdf_paths):
    """
    Creates all required components.
    """

    llm = create_llm()

    memory = create_memory()

    retriever = create_retriever(
        pdf_paths
    )

    return llm, memory, retriever