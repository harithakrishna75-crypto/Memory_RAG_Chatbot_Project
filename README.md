# Memory + RAG Chatbot using Gemini and LangChain

A Retrieval-Augmented Generation (RAG) chatbot built with Google Gemini, LangChain, Chroma, and Streamlit that answers questions from PDF documents while maintaining conversational memory.

## Features

- PDF-based question answering
- Retrieval-Augmented Generation (RAG)
- Semantic search using embeddings
- Chroma vector database for document retrieval
- Conversation memory using LangChain
- Streamlit-based chat interface
- Session reset functionality
- Secure API key management using environment variables

## Tech Stack

- Python
- Google Gemini 2.5 Flash
- LangChain
- Chroma Vector Database
- Google Generative AI Embeddings
- Streamlit
- python-dotenv

## How It Works

The chatbot loads a PDF document, splits it into chunks, converts the chunks into vector embeddings, and stores them in a Chroma vector database. When a user asks a question, the system retrieves the most relevant document sections and provides them to Gemini for answer generation. ConversationSummaryBufferMemory maintains context across interactions, enabling more natural multi-turn conversations.
