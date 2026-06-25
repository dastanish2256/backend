import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from pptx import Presentation
import google.generativeai as genai
import json

app = FastAPI()

# Configure Gemini
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash')

# ── Pydantic Models ──────────────────────────────────────────────────────────

class RAGRequest(BaseModel):
    """Request model for RAG queries"""
    query: str
    documents: List[str] = []
    top_k: int = 5

class RAGSourceDocument(BaseModel):
    """Source document reference"""
    title: str
    content: str
    score: float
    knowledge_base_id: int = 0

class RAGResponse(BaseModel):
    """Response model for RAG queries"""
    answer: str
    sources: List[RAGSourceDocument] = []
    confidence: float = 0.8

class QuizQuestion(BaseModel):
    """Single quiz question"""
    question: str
    options: List[str]
    correct_answer: str

class QuizResponse(BaseModel):
    """Quiz generation response"""
    quiz: List[QuizQuestion]

# ── Helper Functions ────────────────────────────────────────────────────────

def extract_text_from_pptx(file_path: str) -> str:
    """Extract text from PPTX file"""
    prs = Presentation(file_path)
    full_text = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                full_text.append(shape.text)
    return "\n".join(full_text)

def simple_relevance_score(query: str, document: str) -> float:
    """
    Simple relevance scoring based on keyword overlap.
    In production, use semantic similarity (embeddings).
    """
    query_words = set(query.lower().split())
    doc_words = set(document.lower().split())
    
    if len(doc_words) == 0:
        return 0.0
    
    overlap = len(query_words & doc_words)
    score = overlap / len(doc_words)
    return min(score, 1.0)

def rank_documents_by_relevance(query: str, documents: List[str], top_k: int = 5) -> List[RAGSourceDocument]:
    """
    Rank documents by relevance to query and return top_k.
    """
    ranked = []
    for i, doc in enumerate(documents):
        score = simple_relevance_score(query, doc)
        ranked.append(RAGSourceDocument(
            title=f"Document {i+1}",
            content=doc[:500],  # Truncate for response
            score=score,
            knowledge_base_id=i
        ))
    
    # Sort by score descending
    ranked.sort(key=lambda x: x.score, reverse=True)
    return ranked[:top_k]

# ── RAG Endpoints ───────────────────────────────────────────────────────────

@app.post("/api/rag/query", response_model=RAGResponse)
async def rag_query(request: RAGRequest):
    """
    Query RAG service with documents.
    Returns AI-generated answer based on provided documents + referenced sources.
    """
    try:
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Rank documents by relevance
        relevant_docs = rank_documents_by_relevance(
            request.query, 
            request.documents, 
            request.top_k
        )
        
        # Prepare context for Gemini
        context = "\n\n---\n\n".join([doc.content for doc in relevant_docs])
        
        # Construct prompt for Gemini
        prompt = f"""You are a helpful healthcare administrative assistant.
        
Based on the following knowledge base documents, answer the user's question accurately and concisely.

If the answer is not in the documents, say "I don't have information about this in my knowledge base. Please contact support."

KNOWLEDGE BASE:
{context}

USER QUESTION:
{request.query}

INSTRUCTIONS:
- Answer should be clear and professional
- Keep response to 2-3 sentences maximum
- If relevant documents exist, reference them in your answer
- Do not provide medical diagnoses or clinical advice
"""
        
        # Call Gemini API
        response = model.generate_content(prompt)
        answer = response.text
        
        # Calculate overall confidence based on average document scores
        avg_confidence = sum(doc.score for doc in relevant_docs) / len(relevant_docs) if relevant_docs else 0.5
        
        return RAGResponse(
            answer=answer,
            sources=relevant_docs,
            confidence=min(avg_confidence, 1.0)
        )
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# ── Quiz Generation Endpoint (Original) ─────────────────────────────────────

@app.post("/generate-quiz", response_model=QuizResponse)
async def generate_quiz(file: UploadFile = File(...)):
    """
    Generate a 20-question multiple choice quiz from a PowerPoint file.
    """
    if not file.filename.endswith('.pptx'):
        raise HTTPException(status_code=400, detail="Only .pptx files are supported.")
    
    # Save temp file to parse
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
    
    try:
        # 1. Parse PPTX
        context_text = extract_text_from_pptx(temp_path)
        
        # 2. Construct Prompt
        prompt = f"""Extract the most important information from the following text and create a quiz.
Requirements:
- Exactly 20 Multiple Choice Questions
- 4 options per question
- Clearly indicate the correct answer
- Return the result strictly as a JSON array with this structure:
[
  {{
    "question": "question text",
    "options": ["option1", "option2", "option3", "option4"],
    "correct_answer": "option1"
  }},
  ...
]

Text: {context_text[:10000]}"""
        
        # 3. Call Gemini
        response = model.generate_content(prompt)
        
        # Clean up the AI response (removing markdown code blocks if present)
        raw_text = response.text.replace('```json', '').replace('```', '').strip()
        quiz_data = json.loads(raw_text)
        
        return {"quiz": quiz_data}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ── Health Check ────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    """Health check endpoint"""
    return {"status": "Service is running"}

@app.get("/api/health")
def api_health_check():
    """API health check endpoint"""
    return {"status": "running", "version": "1.0.0"}
