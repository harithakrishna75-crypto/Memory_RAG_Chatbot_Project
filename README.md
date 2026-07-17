# Memory + Multi-Document RAG Chatbot using Gemini, LangChain, Chroma, and Hugging Face

A Retrieval-Augmented Generation (RAG) chatbot built with Google Gemini, LangChain, ChromaDB, Hugging Face Embeddings, and Streamlit that enables users to upload multiple PDF documents, create a dynamic knowledge base, retrieve information using hybrid search, and ask natural language questions while maintaining conversational memory.

## Features

### Current Features

* Multi-document PDF upload support
* Retrieval-Augmented Generation (RAG)
* Hybrid Search (BM25 + Vector Search)
* Semantic search using Hugging Face embeddings
* Chroma vector database for document retrieval
* Conversational memory using LangChain
* Streamlit-based chat interface
* Session reset functionality
* Secure API key management using environment variables
* Dynamic knowledge base creation from uploaded PDFs
* Metadata tracking for source documents and page numbers
* Source citations with document name and page number
* Multi-document retrieval across uploaded PDFs
* Context-aware question answering using Gemini

## Tech Stack

* Python
* Google Gemini 2.5 Flash
* LangChain
* Hugging Face Embeddings (all-MiniLM-L6-v2)
* ChromaDB
* Streamlit
* PyPDF
* Python-dotenv
* BM25 Retrieval (rank-bm25)
* Sentence Transformers

## How It Works

1. Users upload one or more PDF documents.
2. Documents are loaded and metadata is attached (document name and page number).
3. Documents are split into manageable chunks.
4. Each chunk is converted into vector embeddings using Hugging Face Sentence Transformers.
5. Embeddings are stored in a Chroma vector database.
6. A BM25 retriever is created for keyword-based search.
7. A vector retriever is created for semantic search.
8. Hybrid Search combines BM25 retrieval and vector retrieval for improved accuracy.
9. The retriever finds the most relevant document chunks across all uploaded PDFs.
10. Retrieved context and conversation history are provided to Gemini.
11. Gemini generates a context-aware response.
12. Source citations are appended to the response.
13. Conversation memory maintains context across multiple interactions.
