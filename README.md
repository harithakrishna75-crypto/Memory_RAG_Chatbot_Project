# Memory + Multi-Document RAG Chatbot using Gemini, LangChain, and Chroma

A Retrieval-Augmented Generation (RAG) chatbot built with Google Gemini, LangChain, ChromaDB, and Streamlit that enables users to upload multiple PDF documents, create a dynamic knowledge base, and ask natural language questions while maintaining conversational memory.

## Features

### Current Features

* Multi-document PDF upload support
* Retrieval-Augmented Generation (RAG)
* Semantic search using vector embeddings
* Chroma vector database for document retrieval
* Conversational memory using LangChain
* Streamlit-based chat interface
* Session reset functionality
* Secure API key management using environment variables
* Dynamic knowledge base creation from uploaded PDFs
* Metadata tracking for source documents and page numbers

## Tech Stack

* Python
* Google Gemini 2.5 Flash
* LangChain
* ChromaDB
* Streamlit
* PyPDF
* Python-dotenv

## How It Works

1. Users upload one or more PDF documents.
2. Documents are loaded and split into manageable chunks.
3. Each chunk is converted into vector embeddings.
4. Embeddings are stored in a Chroma vector database.
5. User queries are converted into vector representations.
6. The retriever finds the most relevant document chunks.
7. Retrieved context and conversation history are provided to Gemini.
8. Gemini generates a context-aware response.
9. Conversation memory maintains context across multiple interactions.
* Knowledge Base Systems
* Generative AI Application Development
