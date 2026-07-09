Memory + Multi-Document RAG Chatbot using Gemini, LangChain, and Chroma

A Retrieval-Augmented Generation (RAG) chatbot built with Google Gemini, LangChain, ChromaDB, and Streamlit that enables users to upload multiple PDF documents, create a dynamic knowledge base, and ask natural language questions while maintaining conversational memory.

Features
Current Features
Multi-document PDF upload support
Retrieval-Augmented Generation (RAG)
Semantic search using vector embeddings
Chroma vector database for document retrieval
Conversational memory using LangChain
Streamlit-based chat interface
Session reset functionality
Secure API key management using environment variables
Dynamic knowledge base creation from uploaded PDFs
Metadata tracking for source documents and page numbers
Tech Stack
Python
Google Gemini 2.5 Flash
LangChain
ChromaDB
Streamlit
PyPDF
Python-dotenv
Architecture
User Uploads PDFs
        ↓
PyPDFLoader
        ↓
Text Chunking
        ↓
Embeddings
        ↓
Chroma Vector Database
        ↓
Retriever
        ↓
Gemini 2.5 Flash
        ↓
Response Generation
How It Works
Users upload one or more PDF documents.
Documents are loaded and split into manageable chunks.
Each chunk is converted into vector embeddings.
Embeddings are stored in a Chroma vector database.
User queries are converted into vector representations.
The retriever finds the most relevant document chunks.
Retrieved context and conversation history are provided to Gemini.
Gemini generates a context-aware response.
Conversation memory maintains context across multiple interactions.
