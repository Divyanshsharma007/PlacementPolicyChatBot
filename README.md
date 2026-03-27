# NSUT Placement Policy Chatbot — Backend

A RAG chatbot that answers student questions **strictly from the NSUT Placement Policy 2025–26 PDF**.  
Built with **FastAPI · LangChain · LangGraph · OpenAI**.

---

## Architecture

```
Student question
      │
      ▼
┌─────────────────────────────────┐
│          FastAPI  /chat         │
│                                 │
│   LangGraph StateGraph          │
│   ┌─────────────────────────┐   │
│   │  1. retrieve  node      │───┼──► FAISS (PDF chunks, top-5)
│   └────────────┬────────────┘   │
│                ▼                │
│   ┌─────────────────────────┐   │
│   │  2. generate  node      │───┼──► GPT-4o-mini
│   └─────────────────────────┘   │
│                                 │
│   In-memory session store       │
│   (session_id → message list)   │
└─────────────────────────────────┘
      │
      ▼
Policy-grounded answer
```

### How it works

| Step | Node | What happens |
|------|------|-------------|
| 1 | `retrieve` | The student's question is embedded and matched against the FAISS index of PDF chunks. The top-5 most relevant passages are returned with their page numbers. |
| 2 | `generate` | The retrieved passages + the conversation history are sent to GPT-4o-mini with a strict system prompt. If the answer is not in the policy, the model says so and directs the student to T&P contacts. |

---

## Project layout

```
backend/
├── main.py                 ← FastAPI app + LangGraph pipeline
├── requirements.txt        ← Pinned Python dependencies
└── Placement_Policy.pdf    ← Your policy PDF goes here
```

---

## Setup

### Prerequisites

- Python 3.10 or newer
- An [OpenAI API key](https://platform.openai.com/api-keys)

---

### 1 — Create a virtual environment

```bash
cd backend
python -m venv venv

# Activate
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows CMD
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### 3 — Add the PDF

Place `Placement_Policy.pdf` inside the `backend/` folder.  
To use a different path, set the `PDF_PATH` environment variable.

### 4 — Set your OpenAI API key

```bash
# macOS / Linux
export OPENAI_API_KEY="sk-..."

# Windows CMD
set OPENAI_API_KEY=sk-...

# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."
```

### 5 — Start the server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The server starts at `http://localhost:8000`.  
Interactive API docs: `http://localhost:8000/docs`

---

## API reference

### `POST /chat`

Send a question and receive a policy-grounded answer.

**Request body**
```json
{
  "session_id": "student_42",
  "message": "What is the CTC threshold for the Dream category in tech roles?"
}
```

**Response**
```json
{
  "session_id": "student_42",
  "answer": "For Tech roles, the Dream Category threshold is ≥ 12 LPA. Companies offering less than 12 LPA fall under the A++ Category."
}
```

---

### `GET /session/{session_id}`

Check how many messages are stored for a session.

```json
{ "session_id": "student_42", "message_count": 6 }
```

---

### `DELETE /session/{session_id}`

Reset the conversation history for a session.

```json
{ "status": "cleared", "session_id": "student_42" }
```

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI secret key |
| `PDF_PATH` | `Placement_Policy.pdf` | Path to the policy PDF |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI model name |

---

## Connecting a frontend

Any frontend can call `POST /chat` with JSON. Example using `fetch`:

```js
const res = await fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    session_id: 'my-session',
    message: 'Can I apply after receiving a PPO?'
  })
});
const { answer } = await res.json();
```

---

## Customisation

| Goal | Where to change |
|------|-----------------|
| Retrieve more/fewer chunks | `search_kwargs={"k": N}` in `build_retriever()` |
| Adjust chunk size | `chunk_size` in `RecursiveCharacterTextSplitter` |
| Swap vector store | Replace `FAISS` with `Chroma`, `Pinecone`, `Qdrant`, etc. |
| Use a stronger model | Set `LLM_MODEL=gpt-4o` env var |
| Persist sessions | Replace the `sessions` dict with Redis or a database |
| Add streaming | Use `llm.astream()` + FastAPI `StreamingResponse` |

---

## Notes

- The model is instructed to refuse answers not found in the policy and to direct students to **tnpcell@nsitonline.in**.
- Session history is **in-memory** and resets when the server restarts. Use Redis for production.
- The embedding model used is `text-embedding-3-small` (cost-efficient and accurate for this use case).
