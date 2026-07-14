Memory + Multi-Document RAG Chatbot using Gemini, LangChain, Chroma, and Hugging Face

A Retrieval-Augmented Generation (RAG) chatbot built with Google Gemini, LangChain, ChromaDB, Hugging Face Embeddings, and Streamlit that enables users to upload multiple PDF documents, create a dynamic knowledge base, retrieve information using hybrid search, and ask natural language questions while maintaining conversational memory.

Features
Current Features
Multi-document PDF upload support
Retrieval-Augmented Generation (RAG)
Hybrid Search (BM25 + Vector Search)
Semantic search using Hugging Face embeddings
Chroma vector database for document retrieval
Conversational memory using LangChain
Streamlit-based chat interface
Session reset functionality
Secure API key management using environment variables
Dynamic knowledge base creation from uploaded PDFs
Metadata tracking for source documents and page numbers
Source citations with document name and page number
Multi-document retrieval across uploaded PDFs
Context-aware question answering using Gemini
Tech Stack
Python
Google Gemini 2.5 Flash
LangChain
Hugging Face Embeddings (all-MiniLM-L6-v2)
ChromaDB
Streamlit
PyPDF
Python-dotenv
BM25 Retrieval (rank-bm25)
Sentence Transformers
How It Works
Users upload one or more PDF documents.
Documents are loaded and metadata is attached (document name and page number).
Documents are split into manageable chunks.
Each chunk is converted into vector embeddings using Hugging Face Sentence Transformers.
Embeddings are stored in a Chroma vector database.
A BM25 retriever is created for keyword-based search.
A vector retriever is created for semantic search.
Hybrid Search combines BM25 retrieval and vector retrieval for improved accuracy.
The retriever finds the most relevant document chunks across all uploaded PDFs.
Retrieved context and conversation history are provided to Gemini.
Gemini generates a context-aware response.
Source citations are appended to the response.
Conversation memory maintains context across multiple interactions.
