# RAG with Pinecone Chatbot

## Overview

This project is a Retrieval-Augmented Generation (RAG) chatbot built using Python and Pinecone vector database. It allows users to query uploaded documents and retrieve context-aware responses using semantic search.

## Features

* Document-based question answering
* Retrieval-Augmented Generation (RAG)
* Semantic search using vector embeddings
* Pinecone vector database integration
* Context-aware answer retrieval

## Tech Stack

* Python
* LangChain
* Pinecone
* OpenAI API (if used)
* Text document processing

## Project Workflow

1. Load and process input documents
2. Split text into chunks
3. Generate embeddings
4. Store embeddings in Pinecone
5. Retrieve relevant chunks based on user query
6. Generate contextual responses

## Files Included

* `app.py` → Main Python application
* `sample_data.txt` → Sample input document
* `requirements.txt` → Required Python libraries

## How to Run

```bash id="u8yquc"
pip install -r requirements.txt
streamlit run app.py
```

## Use Case

This project demonstrates how Pinecone can be used as a vector database for scalable Retrieval-Augmented Generation applications.

## Future Improvements

* Add PDF support
* Add chat history / memory
* Add multi-document upload
* Deploy as a web app

## Author

H. Suma Srivani
