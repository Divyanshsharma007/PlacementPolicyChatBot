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
import pymupdf4llm
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
#from langchain_google_genai import GoogleGenerativeAIEmbeddings
#from langchain_ollama import OllamaLLM
import time
from langchain_core.documents import Document
import pymupdf4llm
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
PDF_PATH = os.environ.get("PDF_PATH", "Placement_Policy.pdf")

load_dotenv()

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
        raise FileNotFoundError(
            f"PDF not found at '{pdf_path}'"
        )

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    # Add structured classification table manually
    docs.append(
        Document(
            page_content="""
Company Classification

Tech Roles:
Dream Category >= 12 LPA
A++ Category < 12 LPA

Non-Tech Roles:
Dream Category >= 7 LPA
A++ Category < 7 LPA

Core Roles:
Dream Category >= 6 LPA
A++ Category < 6 LPA
""",
            metadata={"page": 6}
        )
    )
    faq_docs = [
    Document(
        page_content="""
QUESTION: When can I apply for another company after getting an offer?

ANSWER:
Students placed in DREAM category cannot apply for any further companies.

If a student is placed in an A++ Company, he/she can apply in only Dream companies for a second offer only after 75% of the batch is placed.
""",
        metadata={"page": "FAQ"}
    ),

    Document(
        page_content="""
QUESTION: What happens if I reject an offer after declaration of results?

ANSWER:
120 Days ban from all new and existing placement opportunities.
""",
        metadata={"page": "FAQ"}
    ),

    Document(
        page_content="""
QUESTION: How is a Dream company classified?

ANSWER:
Tech Roles:
Dream Category >= 12 LPA

Non-Tech Roles:
Dream Category >= 7 LPA

Core Roles:
Dream Category >= 6 LPA
""",
        metadata={"page": "FAQ"}
    ),

    Document(
        page_content="""
QUESTION: How is an A++ company classified?

ANSWER:
Tech Roles:
A++ Category < 12 LPA

Non-Tech Roles:
A++ Category < 7 LPA

Core Roles:
A++ Category < 6 LPA
""",
        metadata={"page": "FAQ"}
    ),

     Document(
        page_content="""
QUESTION: Who is the Student Head?

ANSWER:
Divyansh Sharma is the Head Student Placement Coordinator
""",
        metadata={"page": "FAQ"}
    ),

    Document(
        page_content="""
QUESTION: Whom should I contact ?

ANSWER:
Students can reach out to Placement Coordinators,
Training and Placement Officials,
or visit the Training and Placement Section during working hours.

Mail: tnpcell@nsitonline.in

""",
        metadata={"page": "FAQ"}
    ),

    Document(
        page_content="""
QUESTION: What are the internship rules?

ANSWER:
Students who have received a 6-month Internship+FTE Opportunity (DREAM and A++) will be allowed to apply in FTE opportunities lying in the DREAM Category only from March 2026 onwards.

Students who have received a 6-month Internship+Performance based PPO Opportunity (DREAM) will be allowed to apply in FTE opportunities lying in the DREAM Category only from March 2026 onwards.

Students who have received a 6-month Internship+Performance based PPO Opportunity (A++) will be allowed to apply in FTE opportunities lying in the A++ and DREAM Category only from March 2026 onwards.
""",
        metadata={"page": 8}
    ),

    Document(
        page_content="""
QUESTION: Will T&P issue a NoC if I have offline subjects during internship?

ANSWER:
No. If a student receives a six-month internship offer and has offline subjects in 8th semester, T&P Section will not issue NoC.
""",
        metadata={"page": 8}
    ),

    Document(
        page_content="""
QUESTION: Can students contact company HRs directly?

ANSWER:
No. Students are not authorized to directly communicate with company HR representatives regarding placement matters unless explicitly instructed by the Training and Placement Section.
""",
        metadata={"page": 13}
    ),

    Document(
        page_content="""
QUESTION: Who is authorized to communicate with company HRs?

ANSWER:
Only Training and Placement Coordinators and authorized T&P officials are permitted to communicate with company HR representatives regarding placement matters.
""",
        metadata={"page": 13}
    ),

    Document(
        page_content="""
QUESTION: Can I share placement information publicly?

ANSWER:
No. Students should not disclose confidential placement-related information including company details, hiring process details, compensation details, or internal placement communications.
""",
        metadata={"page": 13}
    ),

    Document(
        page_content="""
QUESTION: Can I share placement details with students from other colleges?

ANSWER:
No. Placement-related information is confidential and should not be shared outside the university.
""",
        metadata={"page": 13}
    ),

    Document(
        page_content="""
QUESTION: Can I pay company officials for placement assistance?

ANSWER:
No student should pay a single rupee to any Company Officials in lieu of any Training / Placement fees.
""",
        metadata={"page": 13}
    ),

    Document(
        page_content="""
QUESTION: Can T&P modify penalties?

ANSWER:
Training and Placement Section reserves the discretionary authority to change the duration of the penalty and the severity of the punishment depending on the situation.
""",
        metadata={"page": 13}
    ),

    Document(
        page_content="""
QUESTION: What happens if a situation is not covered in the policy?

ANSWER:
The Training and Placement Section reserves discretionary authority to take decisions in situations not explicitly covered by the placement policy.
""",
        metadata={"page": 13}
    ),

    Document(
        page_content="""
QUESTION: Can parents request recommendations from T&P officials?

ANSWER:
Recommendations, concessions and benefits are provided solely at the discretion of the Training and Placement Section as per policy guidelines.
""",
        metadata={"page": 13}
    ),
]

    docs.extend(faq_docs)



    # Better chunking for policy documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=200,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(docs)

    print(f"📄 Total Documents: {len(docs)}")
    print(f"✂️ Total Chunks: {len(chunks)}")

    # Embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )

    # Build / Load FAISS
    if os.path.exists(FAISS_INDEX_PATH):

        print("📂 Loading existing FAISS index...")

        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    else:

        raise Exception(
        "FAISS index missing. Build locally first."
        )

        vectorstore.save_local(
            FAISS_INDEX_PATH
        )

        print("✅ FAISS index saved.")

    # Retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 12,
            "fetch_k": 30,
            "lambda_mult": 0.7
        }
    )

    return retriever
# Initialize retriever
retriever = None

def get_retriever():
    global retriever

    if retriever is None:
        retriever = build_retriever(PDF_PATH)

    return retriever



# ──────────────────────────────────────────────────────────────────────────────
# LLM
# ──────────────────────────────────────────────────────────────────────────────
#llm = OllamaLLM(
    #model="qwen2.5:3b",
    #temperature=0,
    #top_p=0.01,
    #num_predict=200
    #num_predict=80
#)

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)
# ──────────────────────────────────────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are an NSUT Placement Policy Assistant.
Use ONLY the retrieved policy context.
Important:
- Copy policy facts exactly.
- Preserve all numbers, dates, penalties, and CTC values.
- For classification tables, keep category-value mappings correct.
- If multiple categories exist (Tech, Non-Tech, Core), report all relevant categories instead of selecting one.
- Never expand abbreviations unless explicitly defined in the context.
- Never invent dates, penalties, contacts, or eligibility criteria.
 - Give concise answers.
- Do not explain beyond the retrieved policy text.
- Never define abbreviations.
- Never infer salary ranges.
- Never infer company tiers.
- Never use outside knowledge.
-Use only the retrieved context.
-You may connect a person's name with a designation when both appear together in the retrieved context.
-Do not invent new roles or responsibilities.

If the answer is not present in the context, reply exactly:
This information is not explicitly mentioned in the placement policy.

Context:
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

#def retrieve_context(state: ChatState):

    # Query Expansion
    #expanded_query = f"""
    #Question: {state["question"]}

    #Related placement policy terms:
    #dream company
    #dream category
    #placement offer
    #PPO
    #offer acceptance
    #offer rejection
    #on-campus offer
    #off-campus offer
    #placement rules
    #core role
    #tech role
    #non-tech role
    #CTC category
    #"""
def retrieve_context(state: ChatState):
    start = time.time()

    query = state["question"]
    q = query.lower()

    if any(word in q for word in [
    "rule", "rules", "guideline", "guidelines",
    "discipline", "conduct"
 ]):
    # Query Expansion
        query += """
        placement policy
        eligibility criteria
        student responsibilities
        code of conduct
        disciplinary action
        placement process
        guidelines
        regulations
        """

    if "ppo" in q:
        query += """
        pre placement offer
        ppo
        offer acceptance
        offer rejection
        internship
        """

    if "contact" in q or "whom" in q:
        query += """
        grievance
        complaint
        coordinator
        placement cell
        t&p section
        """

    if "penalty" in q:
        query += """
        penalty structure
        disciplinary action
        offer rejection
        ban period
        """
    if "dream" in q:
        query += """
        dream company
        dream category
        company classification
        tech roles
        non-tech roles
        core roles
        """

    if "a++" in q:
        query += """
        a++ company
        a++ category
        company classification
        tech roles
        non-tech roles
        core roles
        """
    docs = get_retriever().invoke(query)

    print(f"🔍 Retrieval Time: {time.time() - start:.2f}s")

    for i, doc in enumerate(docs):
        print(f"\n--- DOCUMENT {i+1} ---")
        print(doc.page_content[:1000])

    context = "\n\n".join(
        [
            f"[PAGE {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
            for doc in docs
        ]
    )

    return {"context": context}
# ──────────────────────────────────────────────────────────────────────────────
# Generate Answer
# ──────────────────────────────────────────────────────────────────────────────
def generate_answer(state: ChatState):
    chain = prompt_template | llm
    start = time.time()
    response = chain.invoke({
        "context": state["context"],
        "question": state["question"],
    })
    print(f"🤖 LLM Time: {time.time() - start:.2f}s")
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
        "model": "gemini-2.5-flash"
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