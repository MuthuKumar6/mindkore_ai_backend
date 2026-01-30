"""
Gemini AI Agent - FastAPI Server with Authentication + MongoDB
Default model: gemini-2.5-flash

Installation:
pip install fastapi uvicorn google-generativeai python-multipart python-dotenv pyjwt passlib[bcrypt] tenacity motor pydantic[email]

Run:
uvicorn main:app --reload --port 8001
"""

import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, ContentType
from dotenv import load_dotenv
import jwt
from passlib.context import CryptContext
from tenacity import retry, stop_after_attempt, wait_exponential

# Async MongoDB
import motor.motor_asyncio
from bson import ObjectId

# ────────────────────────────────────────────────
#  Configuration
# ────────────────────────────────────────────────

load_dotenv()

SECRET_KEY = os.getenv('SECRET_KEY', 'this-is-not-secure-change-me-in-production')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
MONGODB_URL = os.getenv('MONGODB_URL', 'mongodb://localhost:27017')
DB_NAME = os.getenv('DB_NAME', 'gemini_agent')

if not GEMINI_API_KEY:
    print("⚠️  GEMINI_API_KEY environment variable is missing!")

try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"❌ Gemini configuration failed: {e}")

# MongoDB
mongo_client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URL)
db = mongo_client[DB_NAME]
users_collection = db["users"]
chats_collection = db["chat_sessions"]

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# FastAPI app
app = FastAPI(
    title="Gemini AI Agent API",
    description="FastAPI + Google Gemini 2.5-flash + MongoDB persistent chat + JWT auth",
    version="2.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
origins = [
    "https://mindkoreai.netlify.app",
    "https://*.netlify.app",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "https://mindkore-4b2pmfcdv-muthukumar6s-projects.vercel.app/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https?://.*\.netlify\.app",
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["Content-Type", "Authorization"],
    expose_headers=["Content-Disposition"],
    max_age=600,
)

# ────────────────────────────────────────────────
#  Pydantic Models
# ────────────────────────────────────────────────

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

class UserOut(BaseModel):
    email: str
    name: str
    created_at: datetime

class GenerateRequest(BaseModel):
    prompt: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    model: Optional[str] = "gemini-2.5-flash"

class GenerateResponse(BaseModel):
    success: bool
    text: str
    model: str

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

# ────────────────────────────────────────────────
#  Simple in-memory rate limiter (can be replaced with Redis)
# ────────────────────────────────────────────────

from collections import defaultdict, deque
from contextlib import contextmanager
import time

class BetterRateLimiter:
    def __init__(self):
        self.user_requests: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
        self.global_lock_until = 0.0

    @contextmanager
    def for_user(self, email: str):
        now = time.time()

        if now < self.global_lock_until:
            time.sleep(self.global_lock_until - now + 0.4)

        dq = self.user_requests[email]
        if len(dq) >= 10:
            oldest = dq[0]
            elapsed = now - oldest
            if elapsed < 60:
                wait_time = (60 - elapsed) / (len(dq) - 9) + 0.8
                time.sleep(wait_time)

        dq.append(now)

        try:
            yield
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                self.global_lock_until = max(self.global_lock_until, now + 120)
            raise

rate_limiter = BetterRateLimiter()

# ────────────────────────────────────────────────
#  Auth Helpers
# ────────────────────────────────────────────────

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def create_jwt_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(cred: HTTPAuthorizationCredentials = Depends(security)):
    token = cred.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if not email:
            raise HTTPException(401, "Invalid token")
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token has expired")
    except jwt.JWTError:
        raise HTTPException(401, "Invalid token")

    user = await users_collection.find_one({"email": email})
    if not user:
        raise HTTPException(401, "User not found")

    return user

# ────────────────────────────────────────────────
#  Gemini Helpers
# ────────────────────────────────────────────────

@retry(reraise=True, stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1.5, min=4, max=90))
async def generate_text(prompt: str, temp: float, max_tokens: int, model: str) -> str:
    try:
        m = genai.GenerativeModel(model_name=model)
        resp = await m.generate_content_async(
            [prompt],
            generation_config=GenerationConfig(
                temperature=temp,
                max_output_tokens=max_tokens,
                top_p=0.95,
                top_k=64
            )
        )
        return resp.text
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            raise HTTPException(429, "Rate limit / quota exceeded. Try again later.")
        raise HTTPException(500, f"Generation failed: {str(e)}")

# ────────────────────────────────────────────────
#  Chat Persistence Helpers
# ────────────────────────────────────────────────

async def load_or_create_chat(user_email: str, session_id: str, model_name: str):
    session_key = f"{user_email}_{session_id}"

    doc = await chats_collection.find_one({"session_key": session_key})

    history = []
    if doc and "history" in doc:
        for entry in doc["history"]:
            history.append({
                "role": entry["role"],
                "parts": [{"text": entry["content"]}]
            })

    gem_model = genai.GenerativeModel(model_name=model_name)
    chat = gem_model.start_chat(history=history)

    return chat, session_key, bool(doc)

async def persist_chat_history(session_key: str, history: List[ContentType], model: str):
    messages = []
    for msg in history:
        role = msg.role
        content = msg.parts[0].text if msg.parts else ""
        messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })

    await chats_collection.update_one(
        {"session_key": session_key},
        {
            "$set": {
                "session_key": session_key,
                "user_email": session_key.split("_", 1)[0],
                "model": model,
                "history": messages,
                "last_updated": datetime.utcnow()
            }
        },
        upsert=True
    )

# ────────────────────────────────────────────────
#  Routes - Auth
# ────────────────────────────────────────────────

@app.post("/api/auth/signup", response_model=Token)
async def signup(data: UserSignup):
    if await users_collection.find_one({"email": data.email}):
        raise HTTPException(400, "Email already registered")

    user = {
        "email": data.email,
        "name": data.name,
        "password": hash_password(data.password),
        "created_at": datetime.utcnow()
    }
    await users_collection.insert_one(user)

    token = create_jwt_token({"sub": data.email})

    print("token", token)

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "email": user["email"],
            "name": user["name"],
            "created_at": user["created_at"].isoformat()
        }
    }

@app.post("/api/auth/login", response_model=Token)
async def login(cred: UserLogin):
    user = await users_collection.find_one({"email": cred.email})
    if not user or not verify_password(cred.password, user["password"]):
        raise HTTPException(401, "Invalid email or password")

    token = create_jwt_token({"sub": user["email"]})

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "email": user["email"],
            "name": user["name"],
            "created_at": user["created_at"].isoformat()
        }
    }

@app.get("/api/auth/me", response_model=UserOut)
async def get_current_user_info(user = Depends(get_current_user)):
    return UserOut(
        email=user["email"],
        name=user["name"],
        created_at=user["created_at"]
    )

# ────────────────────────────────────────────────
#  Core Endpoints
# ────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "Gemini AI Agent API",
        "default_model": "gemini-2.5-flash",
        "auth": "JWT required for most endpoints",
        "storage": "MongoDB"
    }

@app.post("/api/generate", response_model=GenerateResponse)
async def generate_text_endpoint(req: GenerateRequest, user = Depends(get_current_user)):
    with rate_limiter.for_user(user["email"]):
        text = await generate_text(
            req.prompt,
            req.temperature,
            req.max_tokens,
            req.model
        )
        return GenerateResponse(
            success=True,
            text=text,
            model=req.model
        )

@app.post("/api/chat", response_model=ChatResponse)
async def send_chat_message(req: ChatRequest, user = Depends(get_current_user)):
    chat, session_key, _ = await load_or_create_chat(
        user["email"], req.session_id, req.model
    )

    with rate_limiter.for_user(user["email"]):
        try:
            @retry(reraise=True, stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1.5, min=4, max=90))
            async def send():
                return await chat.send_message_async(req.message)

            response = await send()

            # Persist
            await persist_chat_history(session_key, chat.history, req.model)

            return ChatResponse(
                success=True,
                message=response.text,
                session_id=req.session_id,
                model=req.model
            )

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                raise HTTPException(429, "Rate limit exceeded – please wait")
            raise HTTPException(500, f"Chat failed: {str(e)}")

@app.get("/api/chat/{session_id}/history")
async def get_session_history(session_id: str, user = Depends(get_current_user)):
    key = f"{user['email']}_{session_id}"
    doc = await chats_collection.find_one({"session_key": key})

    if not doc or "history" not in doc:
        return {"session_id": session_id, "history": []}

    return {
        "session_id": session_id,
        "model": doc.get("model", "unknown"),
        "history": [
            {"role": m["role"], "content": m["content"], "timestamp": m.get("timestamp")}
            for m in doc["history"]
        ]
    }

@app.delete("/api/chat/{session_id}/clear")
async def delete_chat_session(session_id: str, user = Depends(get_current_user)):
    key = f"{user['email']}_{session_id}"
    result = await chats_collection.delete_one({"session_key": key})

    if result.deleted_count == 0:
        raise HTTPException(404, "Chat session not found")

    return {"success": True, "message": f"Session {session_id} deleted"}

# Optional: list available models
@app.get("/api/models")
async def list_gemini_models(_ = Depends(get_current_user)):
    try:
        models = [
            m.name.replace("models/", "")
            for m in genai.list_models()
            if "generateContent" in getattr(m, "supported_generation_methods", [])
        ]
        return {
            "available_models": models,
            "recommended": ["gemini-2.5-flash", "gemini-1.5-pro", "gemini-1.5-flash"]
        }
    except Exception as e:
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)