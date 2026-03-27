"""
NSUT Placement Policy Chatbot — Backend (FREE VERSION)
Stack : FastAPI + LangChain + LangGraph + Ollama + HuggingFace
"""

import operator
import os
from typing import Annotated, TypedDict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse   # ✅ NEW
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph

from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings

# ── Config ─────────────────────────────────────────────────────────────────────
PDF_PATH: str = os.environ.get("PDF_PATH", "Placement_Policy.pdf")

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NSUT Placement Policy Chatbot",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Build retriever ────────────────────────────────────────────────────────────
def build_retriever(pdf_path: str):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at '{pdf_path}'")

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 5})


retriever = build_retriever(PDF_PATH)

# ✅ FIX: Use smaller model (IMPORTANT)
llm = Ollama(model="phi3")   # ← changed from llama3

# ── Prompt ─────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an official assistant for NSUT Placement Policy.

Answer ONLY from the given context.
If not found, say:
"I don't have that information in the Placement Policy."

Context:
{context}
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

# ── LangGraph state ────────────────────────────────────────────────────────────
class ChatState(TypedDict):
    messages: Annotated[list, operator.add]
    question: str
    context: str
    answer: str

# ── Nodes ──────────────────────────────────────────────────────────────────────
def retrieve_context(state: ChatState):
    docs = retriever.invoke(state["question"])
    context = "\n\n".join([d.page_content for d in docs])
    return {"context": context}


def generate_answer(state: ChatState):
    history = state["messages"][:-1]

    chain = prompt_template | llm
    response = chain.invoke({
        "context": state["context"],
        "history": history,
        "question": state["question"],
    })

    return {
        "answer": response,
        "messages": [AIMessage(content=response)],
    }

# ── Graph ──────────────────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(ChatState)
    graph.add_node("retrieve", retrieve_context)
    graph.add_node("generate", generate_answer)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


chat_graph = build_graph()

# ── Session store ──────────────────────────────────────────────────────────────
sessions = {}

# ── Schemas ────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    answer: str

# ── Routes ─────────────────────────────────────────────────────────────────────

# ✅ NEW: Serve frontend
@app.get("/")
def serve_ui():
    return FileResponse("index.html")


# Health check
@app.get("/health")
def health():
    return {"status": "ok", "model": "phi3 (local)"}


# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    msg = req.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="Empty message")

    history = sessions.get(req.session_id, [])
    history.append(HumanMessage(content=msg))

    state: ChatState = {
        "messages": history,
        "question": msg,
        "context": "",
        "answer": "",
    }

    try:
        result = chat_graph.invoke(state)
        answer = result["answer"]

        sessions[req.session_id] = result["messages"]

        return ChatResponse(session_id=req.session_id, answer=answer)

    except Exception as e:
        # ✅ Handle memory errors cleanly
        if "memory" in str(e).lower():
            return ChatResponse(
                session_id=req.session_id,
                answer="⚠️ Model ran out of memory. Switch to phi3 or free RAM."
            )

        print("ERROR:", e)
        return ChatResponse(
            session_id=req.session_id,
            answer="❌ Something went wrong. Try again."
        )