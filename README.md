# NSUT Placement Policy Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about the NSUT Placement Policy using a local Large Language Model (LLM), semantic search, and PDF-based knowledge retrieval.

The system combines FastAPI, LangChain, LangGraph, FAISS, HuggingFace Embeddings, and Ollama to provide accurate, context-aware, and policy-grounded responses.

---

## Problem Statement

Placement policies are often lengthy PDF documents that students find difficult to navigate. Finding specific rules, eligibility criteria, penalties, internship policies, or placement procedures can be time-consuming.

This project converts the placement policy document into an intelligent conversational assistant that can answer questions in natural language while grounding responses in the official policy document.

---

## Features

* PDF-based Knowledge Retrieval
* Retrieval-Augmented Generation (RAG)
* Semantic Search using Embeddings
* Local LLM Inference using Ollama
* Session-based Conversations
* FastAPI REST Backend
* LangGraph Workflow Management
* FAISS Vector Database
* Fully Offline Deployment
* Lightweight and Cost-Free Operation
* Frontend Integration Support
* Policy-Grounded Responses
* Reduced Hallucinations

---

## Tech Stack

### Backend

* Python
* FastAPI
* Uvicorn

### LLM Framework

* LangChain
* LangGraph

### Large Language Models

* Phi-3 Mini
* Qwen2.5
* TinyLlama
* Gemma

### Embeddings

* sentence-transformers/all-MiniLM-L6-v2

### Vector Database

* FAISS

### Document Processing

* PyPDFLoader
* RecursiveCharacterTextSplitter

### Frontend

* HTML
* CSS
* JavaScript

---

## System Architecture

```text
User Question
      │
      ▼
Frontend UI
      │
      ▼
FastAPI Backend
      │
      ▼
LangGraph Workflow
      │
 ┌────┴────┐
 │         │
 ▼         ▼
Retriever  Chat History
 │
 ▼
FAISS Vector Store
 │
 ▼
Relevant Chunks
 │
 ▼
Prompt Builder
 │
 ▼
Ollama LLM
 │
 ▼
Generated Answer
 │
 ▼
Frontend UI
```

---

## How It Works

### Step 1: PDF Loading

The placement policy PDF is loaded using PyPDFLoader.

### Step 2: Chunking

The document is split into smaller chunks for efficient retrieval.

```python
chunk_size = 600
chunk_overlap = 100
```

### Step 3: Embedding Generation

Each chunk is converted into vector embeddings using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

### Step 4: Vector Storage

Embeddings are stored inside a FAISS vector database.

### Step 5: User Query

The student asks a question.

Example:

```text
What happens if I reject an offer after the results are declared?
```

### Step 6: Semantic Retrieval

The most relevant chunks are retrieved using vector similarity search.

### Step 7: Prompt Construction

Retrieved chunks are combined with the user's question.

### Step 8: Response Generation

The Ollama-hosted LLM generates an answer strictly based on the retrieved context.

---

## Project Structure

```text
NSUT-Placement-Chatbot/
│
├── main.py
├── index.html
├── Placement_Policy.pdf
├── requirements.txt
├── README.md
│
├── faiss_index/
│
├── screenshots/
│   ├── home.png
│   ├── chat.png
│
└── assets/
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/nsut-placement-chatbot.git

cd nsut-placement-chatbot
```

---

### Create Virtual Environment

Windows

```bash
python -m venv venv

venv\Scripts\activate
```

Linux / Mac

```bash
python3 -m venv venv

source venv/bin/activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Install Ollama

Download Ollama:

https://ollama.com

Install a model:

```bash
ollama pull phi3
```

or

```bash
ollama pull qwen2.5:1.5b
```

Start Ollama:

```bash
ollama serve
```

---

## Running the Project

Start FastAPI Server:

```bash
uvicorn main:app --reload
```

Application:

```text
http://localhost:8000
```

Health Endpoint:

```text
http://localhost:8000/health
```

---

## API Documentation

### Health Check

```http
GET /health
```

Response

```json
{
  "status": "ok",
  "model": "phi3"
}
```

---

### Chat Endpoint

```http
POST /chat
```

Request

```json
{
  "session_id": "student123",
  "message": "What is the placement eligibility criteria?"
}
```

Response

```json
{
  "session_id": "student123",
  "answer": "..."
}
```

---

## Example Questions

Students can ask:

```text
What is the placement eligibility criteria?

What happens if I reject an offer?

Can I sit for another company after being placed?

What is the ban period for rejecting an offer?

How many offers can a student hold?

What are the internship conversion rules?

What is the placement withdrawal policy?

Are backlogs allowed during placements?
```

---

## Sample Workflow

### User

```text
What happens if I reject an offer after results are declared?
```

### Retrieval

Relevant policy chunks are retrieved from the FAISS database.

### Generation

The LLM uses retrieved context to generate an answer.

### Output

```text
Students rejecting an offer after declaration of results are subject to a 120-day ban from all new and existing placement opportunities.
```

---

## Performance Optimizations

### Optimized Chunking

```python
chunk_size = 600
chunk_overlap = 100
```

### Reduced Retrieval Size

```python
search_kwargs = {
    "k": 4
}
```

### Limited Chat History

```python
history = state["messages"][-4:]
```

### Faster Model Configuration

```python
llm = Ollama(
    model="phi3",
    temperature=0,
    num_predict=180
)
```

### Persistent FAISS Index

The vector database is saved locally and reused across restarts to avoid recomputing embeddings.

Benefits:

* Faster startup
* Reduced memory consumption
* Improved response time

---

## Performance Benchmarks

| Model        | Average Response Time |
| ------------ | --------------------- |
| Llama 3      | 15–30 sec             |
| Phi3 Mini    | 3–8 sec               |
| Qwen2.5 1.5B | 2–5 sec               |
| TinyLlama    | 1–3 sec               |

---

## Why RAG?

Traditional LLMs rely on pretrained knowledge and may hallucinate.

RAG improves reliability by:

* Retrieving actual policy content
* Grounding answers in source documents
* Reducing hallucinations
* Improving factual accuracy
* Supporting document-specific queries

---

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]
```

### Build Docker Image

```bash
docker build -t placement-chatbot .
```

### Run Container

```bash
docker run -p 8000:8000 placement-chatbot
```

---

## Deploy on Render

### Build Command

```bash
pip install -r requirements.txt
```

### Start Command

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

## Deploy on Railway

1. Push repository to GitHub.
2. Create Railway Project.
3. Connect GitHub Repository.
4. Configure environment variables.
5. Deploy.

---

## Future Enhancements

### Version 2.0

* Source Citations
* Streaming Responses
* Multi-PDF Support
* Admin Upload Panel
* Authentication System
* Persistent Chat Memory
* User Analytics

### Version 3.0

* Hybrid Search (BM25 + FAISS)
* Query Rewriting
* Semantic Cache
* Feedback Learning
* Voice Input
* Voice Output
* Dashboard Analytics

---

## Skills Demonstrated

### Artificial Intelligence

* Retrieval-Augmented Generation
* Semantic Search
* Prompt Engineering
* Context Management

### Machine Learning

* Sentence Transformers
* Embedding Models
* Vector Similarity Search

### Backend Development

* FastAPI
* REST APIs
* Session Management

### LLM Engineering

* LangChain
* LangGraph
* Ollama
* Local Model Deployment

### Databases

* FAISS Vector Store

---

## Resume Description

Developed a Retrieval-Augmented Generation (RAG) chatbot for NSUT Placement Policy using FastAPI, LangChain, LangGraph, FAISS, HuggingFace Embeddings, and Ollama. Implemented semantic document retrieval, vector search, and local LLM inference to deliver accurate policy-based responses while reducing hallucinations and enabling fully offline deployment.

---

## Screenshots

Add screenshots in the screenshots folder:

```markdown
![Home Page](screenshots/home.png)

![Chat Interface](screenshots/chat.png)

![Response Example](screenshots/response.png)
```

---

## License

This project is developed for educational and academic purposes.

Feel free to modify and extend the project according to your requirements.

---

## Author

Divyansh Sharma

B.Tech Electronics & Communication Engineering (AI & ML)

Netaji Subhas University of Technology (NSUT)

---

## Acknowledgements

Special thanks to the following open-source technologies:

* [FastAPI](https://fastapi.tiangolo.com?utm_source=chatgpt.com)
* [LangChain](https://www.langchain.com?utm_source=chatgpt.com)
* [LangGraph](https://langchain-ai.github.io/langgraph/?utm_source=chatgpt.com)
* [Ollama](https://ollama.com?utm_source=chatgpt.com)
* [Hugging Face](https://huggingface.co?utm_source=chatgpt.com)
* [FAISS](https://faiss.ai?utm_source=chatgpt.com)
* [Sentence Transformers](https://www.sbert.net?utm_source=chatgpt.com)
