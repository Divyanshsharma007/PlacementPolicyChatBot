"""
NSUT Placement Policy Chatbot — Backend
Tech Stack:
FastAPI + LangChain + LangGraph + Ollama + FAISS + HuggingFace
"""

# ──────────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────────

import os
import asyncio
import operator
from typing import Annotated, TypedDict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Document Loader
from langchain_community.document_loaders import PyPDFLoader

# Vector Database
from langchain_community.vectorstores import FAISS

# Text Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Prompt Template
from langchain_core.prompts import ChatPromptTemplate

# Messages
from langchain_core.messages import AIMessage

# LangGraph
from langgraph.graph import END, StateGraph

# Modern LangChain Imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

PDF_PATH = os.environ.get("PDF_PATH", "Placement_Policy.pdf")

FAISS_INDEX_PATH = "faiss_index"

# ──────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NSUT Placement Policy Chatbot",
    version="2.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────────
# Build Retriever
# ──────────────────────────────────────────────────────────────────────────────

def build_retriever(pdf_path: str):

    # Check PDF exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at '{pdf_path}'")

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Better chunking for policy documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=250,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(docs)

    # Embedding Model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load existing FAISS index if present
    if os.path.exists(FAISS_INDEX_PATH):

        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    else:
        # Create FAISS index
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # Save locally
        vectorstore.save_local(FAISS_INDEX_PATH)

    # Retriever using MMR
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 15,
            "lambda_mult": 0.7
        }
    )

    return retriever


# Initialize retriever
retriever = build_retriever(PDF_PATH)

# ──────────────────────────────────────────────────────────────────────────────
# LLM
# ──────────────────────────────────────────────────────────────────────────────

llm = OllamaLLM(
    model="phi3",
    temperature=0,
    top_p=0.1
)

# ──────────────────────────────────────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are an NSUT Placement Policy retrieval assistant.

You MUST answer ONLY from the retrieved context.

STRICT RULES:
1. Never use outside knowledge.
2. Never give advice or recommendations.
3. Never suggest consulting officials, coordinators, or manuals.
4. Never mention sections unless explicitly present in context.
5. Never infer missing information.
6. Never explain beyond retrieved text.
7. If the answer is not found in the retrieved context, reply EXACTLY:
"This information is not explicitly mentioned in the provided policy context."
8. Keep responses concise and factual.
9. Prefer exact wording from the context.
10. Do not add introductory or concluding sentences.
11. Use exact wording from the retrieved context whenever possible.
12. Do not summarize or reinterpret policy clauses.

Retrieved Context:
{context}
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])

# ──────────────────────────────────────────────────────────────────────────────
# LangGraph State
# ──────────────────────────────────────────────────────────────────────────────

class ChatState(TypedDict):
    messages: Annotated[list, operator.add]
    question: str
    context: str
    answer: str

# ──────────────────────────────────────────────────────────────────────────────
# Retrieve Context
# ──────────────────────────────────────────────────────────────────────────────

def retrieve_context(state: ChatState):

    # Query Expansion
    expanded_query = f"""
    Question: {state["question"]}

    Related placement policy terms:
    dream company
    dream category
    placement offer
    PPO
    offer acceptance
    offer rejection
    on-campus offer
    off-campus offer
    placement rules
    core role
    tech role
    non-tech role
    CTC category
    """

    # Retrieve docs
    docs = retriever.invoke(expanded_query)

    # DEBUGGING
    for i, d in enumerate(docs):
        print(f"\n--- DOCUMENT {i+1} ---\n")
        print(d.page_content[:800])

    # Merge context
    context = "\n\n".join([d.page_content for d in docs])

    return {"context": context}

# ──────────────────────────────────────────────────────────────────────────────
# Generate Answer
# ──────────────────────────────────────────────────────────────────────────────

def generate_answer(state: ChatState):

    chain = prompt_template | llm

    response = chain.invoke({
        "context": state["context"],
        "question": state["question"],
    })

    # Extract response safely
    answer_text = (
        response.content
        if hasattr(response, "content")
        else str(response)
    )

    # Cleanup
    answer_text = answer_text.strip()
    answer_text = answer_text.replace(". ", ".\n")

    # Hallucination filtering
    forbidden_phrases = [
        "consult",
        "please refer",
        "placement coordinator",
        "knowledge cutoff",
        "it may vary",
    ]

    if any(p in answer_text.lower() for p in forbidden_phrases):
        answer_text = (
            "This information is not explicitly mentioned "
            "in the provided policy context."
        )

    return {
        "answer": answer_text,
        "messages": [AIMessage(content=answer_text)],
    }

# ──────────────────────────────────────────────────────────────────────────────
# Build LangGraph
# ──────────────────────────────────────────────────────────────────────────────

def build_graph():

    graph = StateGraph(ChatState)

    graph.add_node("retrieve", retrieve_context)
    graph.add_node("generate", generate_answer)

    graph.set_entry_point("retrieve")

    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


chat_graph = build_graph()

# ──────────────────────────────────────────────────────────────────────────────
# API Schemas
# ──────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str

# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

# Serve frontend
@app.get("/")
def serve_ui():

    if os.path.exists("index.html"):
        return FileResponse("index.html")

    return {"message": "Frontend not found"}

# Health Check
@app.get("/health")
def health():

    return {
        "status": "ok",
        "model": "phi3"
    }

# Chat Endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):

    msg = req.message.strip()

    if not msg:
        raise HTTPException(
            status_code=400,
            detail="Empty message"
        )

    # Build state
    state: ChatState = {
        "messages": [],
        "question": msg,
        "context": "",
        "answer": "",
    }

    try:

        # Run graph in separate thread
        result = await asyncio.to_thread(
            chat_graph.invoke,
            state
        )

        answer = result["answer"]

        return ChatResponse(
            session_id=req.session_id,
            answer=answer
        )

    except Exception as e:

        print("ERROR:", e)

        # Memory errors
        if "memory" in str(e).lower():
            return ChatResponse(
                session_id=req.session_id,
                answer="⚠️ Model ran out of memory."
            )

        return ChatResponse(
            session_id=req.session_id,
            answer="❌ Something went wrong."
        )