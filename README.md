FastAPI Chatbot
---------------

A FastAPI-based chatbot that uses Gemini, ChromaDB, and LangChain to provide question answering and information retrieval.

Features
--------
- FastAPI backend
- Retrieval-Augmented Generation (RAG) using ChromaDB
- Google Gemini for text generation
- Supports text and voice inputs
- Easy to deploy and lightweight

How to Run Locally
------------------
Install dependencies:
    pip install -r requirements.txt

Start the server:
    uvicorn app:app --reload

Build Knowledge Base
--------------------
Initialize ChromaDB and generate embeddings:
    python setup_knowledge_base.py

Project Structure
-----------------
app.py
setup_knowledge_base.py
requirements.txt
Dockerfile
data/
chroma_db/

Run with Docker
---------------
Build the Docker image:
    docker build -t chatbot .

Run the container:
    docker run -p 7860:7860 chatbot

Deployment
----------
This project is compatible with Hugging Face Spaces using the Docker template.
