"""
Gemini AI Agent - FastAPI Server
Installation: pip install fastapi uvicorn google-genai python-multipart python-dotenv
Run: uvicorn main:app --reload
"""

import os
import time
from typing import Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="Gemini AI Agent API",
    description="A FastAPI wrapper for Google's Gemini AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class GenerateRequest(BaseModel):
    prompt: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    model: Optional[str] = "gemini-2.5-flash"

class GenerateResponse(BaseModel):
    success: bool
    text: str
    model: str
    tokens_used: Optional[int] = None

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    temperature: Optional[float] = 0.7
    model: Optional[str] = "gemini-2.5-flash"

class ChatResponse(BaseModel):
    success: bool
    message: str
    session_id: str
    model: str

# In-memory storage for chat sessions
chat_sessions = {}

# Rate limiting
class RateLimiter:
    def __init__(self):
        self.last_request_time = 0
        self.request_count = 0
    
    def check_and_wait(self):
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < 4:  # 15 req/min = 1 per 4 sec
            wait_time = 4 - time_since_last
            time.sleep(wait_time)
        
        self.last_request_time = time.time()
        self.request_count += 1

rate_limiter = RateLimiter()

# Initialize Gemini client
API_KEY = os.getenv('GEMINI_API_KEY')

if not API_KEY:
    print("="*60)
    print("⚠️  WARNING: GEMINI_API_KEY not found!")
    print("="*60)
    print("Please create a .env file with:")
    print("GEMINI_API_KEY=your-api-key-here")
    print("="*60)

try:
    gemini_client = genai.Client(api_key=API_KEY)
except Exception as e:
    print(f"❌ Failed to initialize Gemini client: {e}")
    gemini_client = None


# Helper Functions
def generate_text(prompt: str, temperature: float, max_tokens: int, model: str) -> str:
    """Generate text using Gemini"""
    rate_limiter.check_and_wait()
    
    try:
        response = gemini_client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=0.95,
                top_k=40
            )
        )
        return response.text
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            raise HTTPException(status_code=429, detail="Quota exceeded. Please wait and try again.")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Gemini AI Agent API",
        "version": "1.0.0",
        "endpoints": {
            "generate": "/api/generate",
            "chat": "/api/chat",
            "chat_history": "/api/chat/{session_id}/history",
            "clear_chat": "/api/chat/{session_id}/clear",
            "models": "/api/models",
            "health": "/api/health"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "gemini_client": "connected" if gemini_client else "not configured",
        "total_requests": rate_limiter.request_count
    }

@app.post("/api/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text from a prompt"""
    if not gemini_client:
        raise HTTPException(status_code=503, detail="Gemini client not configured")
    
    try:
        text = generate_text(
            prompt=request.prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            model=request.model
        )
        
        return GenerateResponse(
            success=True,
            text=text,
            model=request.model
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the AI (maintains conversation context)"""
    if not gemini_client:
        raise HTTPException(status_code=503, detail="Gemini client not configured")
    
    session_id = request.session_id
    
    # Create new session if doesn't exist
    if session_id not in chat_sessions:
        chat_sessions[session_id] = gemini_client.chats.create(model=request.model)
    
    rate_limiter.check_and_wait()
    
    try:
        response = chat_sessions[session_id].send_message(request.message)
        
        return ChatResponse(
            success=True,
            message=response.text,
            session_id=session_id,
            model=request.model
        )
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            raise HTTPException(status_code=429, detail="Quota exceeded. Please wait and try again.")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.get("/api/chat/{session_id}/history")
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        history = chat_sessions[session_id].history
        return {
            "session_id": session_id,
            "history": [{"role": msg.role, "content": msg.parts[0].text} for msg in history]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/chat/{session_id}/clear")
async def clear_chat(session_id: str):
    """Clear a chat session"""
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        return {"success": True, "message": f"Session {session_id} cleared"}
    raise HTTPException(status_code=404, detail="Session not found")

@app.get("/api/models")
async def list_models():
    """List available Gemini models"""
    if not gemini_client:
        raise HTTPException(status_code=503, detail="Gemini client not configured")
    
    try:
        models = gemini_client.models.list()
        model_list = [model.name for model in models]
        return {
            "models": model_list,
            "recommended": [
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                "gemini-flash-latest"
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats():
    """Get API usage statistics"""
    return {
        "total_requests": rate_limiter.request_count,
        "active_chat_sessions": len(chat_sessions),
        "chat_session_ids": list(chat_sessions.keys())
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)