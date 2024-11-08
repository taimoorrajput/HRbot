from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional, List
import pdfplumber
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
import os
import tempfile
import re
import pickle
from datetime import datetime
import json
from pathlib import Path
import secrets
import uvicorn

# Configuration
STORAGE_DIR = Path("storage")
VECTOR_STORE_PATH = STORAGE_DIR / "vector_store.pkl"
METADATA_PATH = STORAGE_DIR / "metadata.pkl"
GROQ_API_KEY = "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"

# Create storage directory
STORAGE_DIR.mkdir(exist_ok=True)

# Initialize FastAPI app
app = FastAPI(title="Policy Document Chat System")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBasic()
HR_CREDENTIALS = {
    "hr@company.com": "hr123"  # In production, use hashed passwords and proper DB
}

# Initialize embeddings
embeddings = SentenceTransformerEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Active connections store
active_connections: Dict[str, WebSocket] = {}

# Pydantic models
class QuestionRequest(BaseModel):
    question: str

class MetadataResponse(BaseModel):
    upload_date: str
    uploaded_by: str
    filename: str

# Helper functions
def verify_hr_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    email = credentials.username
    password = credentials.password
    if email not in HR_CREDENTIALS or not secrets.compare_digest(
        password, HR_CREDENTIALS[email]
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return email

def minimal_cleaning(text: str) -> str:
    text = re.sub(r'(?<=\w)(?=[A-Z])', ' ', text)
    return ' '.join(text.split())

def process_pdf(pdf_file: UploadFile) -> List[Dict]:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        content = pdf_file.file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    pages = []
    with pdfplumber.open(tmp_file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            raw_text = page.extract_text()
            if raw_text:
                cleaned_text = minimal_cleaning(raw_text)
                pages.append({"page_number": i + 1, "content": cleaned_text})
    
    os.unlink(tmp_file_path)
    return pages

def get_chunks(pages: List[Dict]) -> List[str]:
    documents = "\n\n".join(page["content"] for page in pages)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " "]
    )
    return text_splitter.split_text(documents)

def get_response(question: str, vector_store) -> str:
    question_embedding = embeddings.embed_query(question)
    relevant_docs = vector_store.similarity_search_by_vector(question_embedding, k=4)
    
    context_with_metadata = []
    for doc in relevant_docs:
        page_num = doc.metadata.get("page_number", "unknown")
        content = getattr(doc, 'content', None) or getattr(doc, 'page_content', None)
        if content:
            context_with_metadata.append(f"[Page {page_num}]: {content}")
    
    context = "\n\n".join(context_with_metadata)
    
    llm = ChatGroq(
        model="mixtral-8x7b-32768",
        api_key=GROQ_API_KEY,
        temperature=0.1
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant specializing in answering questions based on company policy documents.
         Analyze the context carefully and provide specific details. Always cite the policy section or page number.
         
         Format your response with:
         1. Direct answer to the question
         2. Supporting details with section references
         
         Context:
         {context}
         """),
        ("human", "{question}")
    ])
    
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.invoke({"context": context, "question": question})
    return response["text"]

# WebSocket connection manager
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    active_connections[client_id] = websocket
    try:
        while True:
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            if not VECTOR_STORE_PATH.exists():
                await websocket.send_text(json.dumps({
                    "error": "No policy document is currently loaded."
                }))
                continue
                
            with open(VECTOR_STORE_PATH, 'rb') as f:
                vector_store = pickle.load(f)
            
            response = get_response(request_data["question"], vector_store)
            await websocket.send_text(json.dumps({
                "response": response
            }))
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
    finally:
        active_connections.pop(client_id, None)

# REST endpoints
@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    email: str = Depends(verify_hr_credentials)  # Enforces HR credentials
):
    try:
        pages = process_pdf(file)
        chunks = get_chunks(pages)
        vector_store = FAISS.from_texts(chunks, embeddings)
        
        metadata = {
            'upload_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'uploaded_by': email,
            'filename': file.filename
        }
        
        with open(VECTOR_STORE_PATH, 'wb') as f:
            pickle.dump(vector_store, f)
        with open(METADATA_PATH, 'wb') as f:
            pickle.dump(metadata, f)
        
        return {"message": "Document processed and saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/document")
async def delete_document(email: str = Depends(verify_hr_credentials)):
    try:
        if VECTOR_STORE_PATH.exists():
            VECTOR_STORE_PATH.unlink()
        if METADATA_PATH.exists():
            METADATA_PATH.unlink()
        return {"message": "Document removed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metadata", response_model=Optional[MetadataResponse])
async def get_metadata():
    try:
        if METADATA_PATH.exists():
            with open(METADATA_PATH, 'rb') as f:
                return pickle.load(f)
        return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)