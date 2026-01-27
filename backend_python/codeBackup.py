

"""
Gemini AI Agent - FastAPI Server with Authentication
Installation: pip install fastapi uvicorn google-generativeai python-multipart python-dotenv pyjwt passlib[bcrypt] tenacity
Run: uvicorn main:app --reload --port 8001
"""

import os
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from dotenv import load_dotenv
import jwt
from passlib.context import CryptContext
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()

# Security
SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-change-this-in-production')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Configure Gemini
API_KEY = os.getenv('GEMINI_API_KEY')

if not API_KEY:
    print("="*60)
    print("⚠️  WARNING: GEMINI_API_KEY not found!")
    print("="*60)

try:
    genai.configure(api_key=API_KEY)
except Exception as e:
    print(f"❌ Failed to configure Gemini: {e}")

# Initialize FastAPI
app = FastAPI(
    title="Gemini AI Agent API with Auth",
    description="A FastAPI wrapper for Google's Gemini AI with authentication",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

origins = [
    "https://mindkoreai.netlify.app",
    "https://*.netlify.app",               # Netlify preview deploys
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https?://.*\.netlify\.app",  # catches all Netlify subdomains
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
    expose_headers=["Content-Disposition"],
    max_age=600,  # Cache preflight responses for 10 minutes
)

# In-memory storage (replace with database in production)
users_db = {}
chat_sessions = {}

# Models
class UserSignup(BaseModel):
    email: EmailStr
    password: str
    name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user: dict

class User(BaseModel):
    email: str
    name: str
    created_at: datetime

class GenerateRequest(BaseModel):
    prompt: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    model: Optional[str] = "gemini-2.5-flash"  # Updated to valid model name

class GenerateResponse(BaseModel):
    success: bool
    text: str
    model: str

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    temperature: Optional[float] = 0.7
    model: Optional[str] = "gemini-2.5-flash"  # Updated to valid model name

class ChatResponse(BaseModel):
    success: bool
    message: str
    session_id: str
    model: str

# Improved Rate Limiter with per-user burst protection and global backoff on 429
class BetterRateLimiter:
    def __init__(self):
        # Per-user: deque of timestamps, maxlen=15 (allow ~8-10 bursts then slow)
        self.user_requests: dict[str, deque] = defaultdict(lambda: deque(maxlen=15))
        self.global_lock_time = 0.0  # Timestamp until which all requests are blocked

    @contextmanager
    def for_user(self, user_email: str):
        now = time.time()

        # Enforce global backoff
        if now < self.global_lock_time:
            remaining = self.global_lock_time - now
            time.sleep(remaining + 0.3)  # Extra buffer

        # Per-user rate check
        user_deque = self.user_requests[user_email]
        if len(user_deque) >= 8:  # Start slowing after 8 requests in window
            oldest = user_deque[0]
            elapsed = now - oldest
            if elapsed < 60:  # 60-second window
                wait = (60 - elapsed) / (len(user_deque) - 7) + 1  # Progressive wait
                time.sleep(wait)

        user_deque.append(now)
        try:
            yield
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                # Increase backoff on quota error
                self.global_lock_time = max(self.global_lock_time, now + 90)
            raise

rate_limiter = BetterRateLimiter()

# Auth functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_token(token)
    email = payload.get("sub")
    
    if email is None or email not in users_db:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    
    return users_db[email]

# Helper Functions
@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=4, max=60)
)
def generate_text(prompt: str, temperature: float, max_tokens: int, model: str) -> str:
    try:
        gemini_model = genai.GenerativeModel(model_name=model)
        response = gemini_model.generate_content(
            [prompt],
            generation_config=GenerationConfig(
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

# Auth Endpoints
@app.post("/api/auth/signup", response_model=Token)
async def signup(user_data: UserSignup):
    """Sign up a new user"""
    if user_data.email in users_db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    hashed_password = get_password_hash(user_data.password)
    
    user = {
        "email": user_data.email,
        "name": user_data.name,
        "password": hashed_password,
        "created_at": datetime.utcnow()
    }
    
    users_db[user_data.email] = user
    
    access_token = create_access_token(data={"sub": user_data.email})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "email": user["email"],
            "name": user["name"],
            "created_at": user["created_at"].isoformat()
        }
    }

@app.post("/api/auth/login", response_model=Token)
async def login(credentials: UserLogin):
    """Login user"""
    if credentials.email not in users_db:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    user = users_db[credentials.email]
    
    if not verify_password(credentials.password, user["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    access_token = create_access_token(data={"sub": user["email"]})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "email": user["email"],
            "name": user["name"],
            "created_at": user["created_at"].isoformat()
        }
    }

@app.get("/api/auth/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    """Get current user info"""
    return {
        "email": current_user["email"],
        "name": current_user["name"],
        "created_at": current_user["created_at"].isoformat()
    }

# Protected Endpoints
@app.get("/")
async def root():
    return {
        "message": "Gemini AI Agent API with Authentication",
        "version": "2.0.0",
        "auth_required": True
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "gemini_configured": True,  # Assuming configured if no error
        "total_requests": len(rate_limiter.user_requests)  # Rough count
    }

@app.post("/api/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest, current_user: dict = Depends(get_current_user)):
    """Generate text from a prompt (protected)"""
    user_email = current_user["email"]
    with rate_limiter.for_user(user_email):
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
async def chat(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    """Chat with AI (protected)"""
    user_email = current_user["email"]
    session_key = f"{user_email}_{request.session_id}"
    
    if session_key not in chat_sessions:
        gemini_model = genai.GenerativeModel(model_name=request.model)
        chat_sessions[session_key] = gemini_model.start_chat(history=[])
    
    with rate_limiter.for_user(user_email):
        try:
            @retry(
                reraise=True,
                stop=stop_after_attempt(4),
                wait=wait_exponential(multiplier=1, min=4, max=60)
            )
            def send_message_with_retry(chat, message):
                return chat.send_message(message)
            
            response = send_message_with_retry(chat_sessions[session_key], request.message)
            
            return ChatResponse(
                success=True,
                message=response.text,
                session_id=request.session_id,
                model=request.model
            )
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                raise HTTPException(status_code=429, detail="Quota exceeded. Please wait and try again.")
            raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.get("/api/chat/{session_id}/history")
async def get_chat_history(session_id: str, current_user: dict = Depends(get_current_user)):
    """Get chat history"""
    user_email = current_user["email"]
    session_key = f"{user_email}_{session_id}"
    
    if session_key not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        history = chat_sessions[session_key].history
        return {
            "session_id": session_id,
            "history": [{"role": msg.role, "content": msg.parts[0].text} for msg in history]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/chat/{session_id}/clear")
async def clear_chat(session_id: str, current_user: dict = Depends(get_current_user)):
    """Clear chat session"""
    user_email = current_user["email"]
    session_key = f"{user_email}_{session_id}"
    
    if session_key in chat_sessions:
        del chat_sessions[session_key]
        return {"success": True, "message": f"Session {session_id} cleared"}
    raise HTTPException(status_code=404, detail="Session not found")

@app.get("/api/models")
async def list_models(current_user: dict = Depends(get_current_user)):
    """List available models"""
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        return {
            "models": models,
            "recommended": ["gemini-2.5-flash", "gemini-1.5-pro"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats(current_user: dict = Depends(get_current_user)):
    """Get user stats"""
    user_email = current_user["email"]
    user_sessions = [key for key in chat_sessions.keys() if key.startswith(user_email)]
    
    return {
        "your_active_sessions": len(user_sessions),
        "total_users": len(users_db)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)