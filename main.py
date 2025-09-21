# main.py
#
# This file contains the complete backend for LinguaLink, a real-time chat application
# built with FastAPI, MongoDB, and integrated with the Gemini API for translation.
#
# Final Version Features (with OAuth fixes):
# - Fixed Google Sign-In with server-side state management (no cookie dependencies)
# - Added a (simulated) welcome email feature for new user registrations.
# - Secure User Authentication via Google OAuth2 Authorization Code Flow.
# - Server configured to run on port 3000.
# - Automatic creation of MongoDB collections and indexes on startup.
# - Real-time messaging with WebSockets.
# - Automatic message translation using the Gemini API.
# - User discovery by email and creation of new chat rooms.
# - Detailed chat history retrieval with participant information.
# - At-rest encryption for all message content and translations.
# - API endpoints for managing user profiles (like default language) and chats.

import os
import json
import asyncio
import secrets
import time
from datetime import datetime
from typing import List, Dict, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, Depends, HTTPException, status, WebSocketDisconnect, Query, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from bson.errors import InvalidId
from jose import jwt, JWTError
from cryptography.fernet import Fernet
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from google_auth_oauthlib.flow import Flow
import httpx
from pymongo import ASCENDING, DESCENDING

# --- Configuration ---
# For production, load these from environment variables or a secure vault.

# MongoDB Configuration
MONGO_DETAILS = os.environ.get("MONGO_DETAILS", "mongodb://localhost:27017")

# Encryption Key for message content
# IMPORTANT: Generate a persistent key for production: `Fernet.generate_key().decode()`
ENCRYPTION_KEY = os.environ.get("ENCRYPTION_KEY", Fernet.generate_key().decode())
fernet = Fernet(ENCRYPTION_KEY.encode())

# Google OAuth2 Configuration
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "660588293356-insq0j2214him7fq448sskegc7frnmhf.apps.googleusercontent.com")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "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")
# This redirect URI MUST be authorized in your Google Cloud Console.
REDIRECT_URI = "http://127.0.0.1:3000/auth/google/callback"
# Allows OAuth2 to work over HTTP for local development. Remove in production (use HTTPS).
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# Gemini API Configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBcnPIGkKdkSpoJaPv3W3mw3uV7c9pH2QI")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# JWT Secret Key for session tokens
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "a_very_secret_key_for_jwt_tokens")
ALGORITHM = "HS256"

# Server-side OAuth state storage (in production, use Redis or database)
oauth_states: Dict[str, float] = {}  # {state: timestamp}

# --- Initialization ---
app = FastAPI(
    title="LinguaLink API",
    description="Backend for a WhatsApp-style chat application with real-time translation.",
    version="1.3.3"  # Updated version with full serialization fix
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Database Connection ---
client = AsyncIOMotorClient(MONGO_DETAILS)
db = client.lingualink_db
users_collection = db.get_collection("users")
chats_collection = db.get_collection("chats")
messages_collection = db.get_collection("messages")

@app.on_event("startup")
async def startup_db_client():
    """On startup, ensure collections and necessary indexes exist."""
    print("Initializing database...")
    await users_collection.create_index("email", unique=True)
    await users_collection.create_index("google_id", unique=True, sparse=True)
    await chats_collection.create_index("participants")
    await messages_collection.create_index([("chat_id", ASCENDING), ("created_at", DESCENDING)])
    print("Database initialization complete.")

@app.on_event("shutdown")
async def shutdown_db_client():
    """Close the database connection on shutdown."""
    client.close()

# --- Pydantic Models (Data Schemas) ---
class User(BaseModel):
    id: str = Field(..., alias="_id")
    google_id: Optional[str] = None
    email: str
    name: str
    picture: Optional[str] = None
    default_language: str = "en"

class UserUpdate(BaseModel):
    default_language: Optional[str] = None

class Message(BaseModel):
    id: str = Field(..., alias="_id")
    chat_id: str
    sender_id: str
    original_content: str
    translations: Dict[str, str] = {}
    created_at: datetime

class ChatParticipant(BaseModel):
    id: str
    name: str
    picture: Optional[str] = None

class ChatDetail(BaseModel):
    id: str = Field(..., alias="_id")
    participants: List[ChatParticipant]
    last_message_content: Optional[str] = None
    last_message_timestamp: Optional[datetime] = None
    created_at: datetime

# --- Authentication & Security ---
def create_access_token(data: dict):
    return jwt.encode(data, JWT_SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Query(...)):
    """Dependency to get the current user from a token."""
    credentials_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub", "")
        if not email:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = await users_collection.find_one({"email": email})
    if user is None:
        raise credentials_exception
    user['_id'] = str(user['_id'])
    return User(**user)

def encrypt_message(text: str) -> str:
    return fernet.encrypt(text.encode()).decode()

def decrypt_message(encrypted_text: str) -> str:
    return fernet.decrypt(encrypted_text.encode()).decode()

# --- OAuth State Management ---
def cleanup_expired_states():
    """Remove expired OAuth states (older than 10 minutes)"""
    current_time = time.time()
    expired_states = [state for state, timestamp in oauth_states.items() 
                     if current_time - timestamp > 600]  # 10 minutes
    for state in expired_states:
        del oauth_states[state]

# --- Email Service (Simulation) ---
async def send_welcome_email(user_info: dict):
    """Simulates sending a welcome email to a new user."""
    # In a real application, you would use a library like `aiosmtplib`
    # or an API service like SendGrid or Mailgun here.
    await asyncio.sleep(1) # Simulate network delay
    print("\n--- Sending Welcome Email (Simulation) ---")
    print(f"To: {user_info['email']}")
    print(f"Subject: Welcome to LinguaLink, {user_info['name']}!")
    print("Body: Thank you for joining our community. Start chatting and connecting across languages!")
    print("----------------------------------------\n")

# --- Gemini Translation Service ---
async def translate_text_gemini(text: str, target_language: str) -> str:
    """Translates text using the Gemini API, with a fallback to the original text."""
    if not GEMINI_API_KEY or not text:
        return text
    
    headers = {"Content-Type": "application/json"}
    prompt = f"Translate the following text to {target_language}. Provide only the translation, without any preamble or explanation: '{text}'"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(GEMINI_API_URL, json=payload, headers=headers, timeout=10.0)
            response.raise_for_status()
            result = response.json()
            translation = result['candidates'][0]['content']['parts'][0]['text']
            return translation.strip()
        except Exception as e:
            print(f"Gemini translation error: {e}")
            return text

# --- WebSocket Connection Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, chat_id: str):
        await websocket.accept()
        if chat_id not in self.active_connections:
            self.active_connections[chat_id] = []
        self.active_connections[chat_id].append(websocket)

    def disconnect(self, websocket: WebSocket, chat_id: str):
        if chat_id in self.active_connections:
            self.active_connections[chat_id].remove(websocket)

    async def broadcast(self, message: str, chat_id: str):
        if chat_id in self.active_connections:
            for connection in self.active_connections[chat_id]:
                await connection.send_text(message)

manager = ConnectionManager()

# --- API Endpoints ---
@app.get("/", response_class=FileResponse)
async def read_root():
    """Serves the main chat UI file."""
    return FileResponse("index.html")

@app.get("/auth/google/signin")
async def auth_google_signin():
    """Initiates the Google Sign-In process with server-side state management."""
    if not os.path.exists("client_secrets.json"):
        raise HTTPException(status_code=500, detail="CRITICAL ERROR: 'client_secrets.json' not found.")
    
    try:
        # Clean up expired states
        cleanup_expired_states()
        
        flow = Flow.from_client_secrets_file(
            client_secrets_file="client_secrets.json",
            scopes=[
                'openid', 
                'https://www.googleapis.com/auth/userinfo.email', 
                'https://www.googleapis.com/auth/userinfo.profile'
            ],
            redirect_uri=REDIRECT_URI
        )
        
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true'
        )
        
        # Store state server-side instead of in cookies
        oauth_states[state] = time.time()
        
        print(f"Generated and stored state: {state}")
        
        return RedirectResponse(authorization_url)
        
    except Exception as e:
        print(f"OAuth signin error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initiate authentication: {str(e)}")

@app.get("/auth/google/callback")
async def auth_google_callback(request: Request):
    """Handles the callback from Google with server-side state validation."""
    
    # Get state from query parameters
    query_state = request.query_params.get("state")
    
    print(f"Received query state: {query_state}")
    print(f"Stored states: {list(oauth_states.keys())}")
    
    # Check if state exists and is valid
    if not query_state:
        raise HTTPException(status_code=401, detail="Missing state parameter (CSRF protection).")
    
    if query_state not in oauth_states:
        raise HTTPException(status_code=401, detail="Invalid or expired state parameter (CSRF protection).")
    
    # Check if state is not expired (10 minutes)
    if time.time() - oauth_states[query_state] > 600:
        del oauth_states[query_state]
        raise HTTPException(status_code=401, detail="Expired state parameter (CSRF protection).")
    
    # Clean up used state
    del oauth_states[query_state]

    # Check for authorization code
    code = request.query_params.get("code")
    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code.")

    if not os.path.exists("client_secrets.json"):
        raise HTTPException(status_code=500, detail="CRITICAL ERROR: 'client_secrets.json' not found.")
        
    try:
        flow = Flow.from_client_secrets_file(
            client_secrets_file="client_secrets.json",
            scopes=None,
            state=query_state,
            redirect_uri=REDIRECT_URI
        )
        
        # Use the full URL string to fetch the token
        authorization_response = str(request.url)
        flow.fetch_token(authorization_response=authorization_response)
        
        credentials = flow.credentials
        
        # Try token verification with increasing clock skew tolerance
        id_info = None
        for skew in [0, 10, 30, 60]:  # Try 0, 10, 30, and 60 seconds tolerance
            try:
                id_info = id_token.verify_oauth2_token(
                    credentials.id_token, 
                    google_requests.Request(), 
                    GOOGLE_CLIENT_ID,
                    clock_skew_in_seconds=skew
                )
                if skew > 0:
                    print(f"Token verified with {skew}s clock skew tolerance")
                break
            except ValueError as e:
                if "too early" in str(e) and skew < 60:
                    continue  # Try with more tolerance
                else:
                    raise e
        
        if not id_info:
            raise HTTPException(status_code=500, detail="Could not verify OAuth token")
        
        email = id_info['email']
        google_id = id_info['sub']

        print(f"Successfully authenticated user: {email}")

        # Find user by google_id, then by email, or create new
        user = await users_collection.find_one({"google_id": google_id})
        if not user:
            user = await users_collection.find_one({"email": email})
            if user:
                # User exists, link their Google ID
                await users_collection.update_one(
                    {"_id": user["_id"]},
                    {"$set": {"google_id": google_id}}
                )
                print(f"Linked Google ID to existing user: {email}")
            else:
                # User does not exist, create a new one
                new_user_data = {
                    "google_id": google_id,
                    "email": email, 
                    "name": id_info.get("name"), 
                    "picture": id_info.get("picture"),
                    "default_language": "en", 
                    "created_at": datetime.utcnow()
                }
                await users_collection.insert_one(new_user_data)
                print(f"Created new user: {email}")
                # Send welcome email in the background
                asyncio.create_task(send_welcome_email(new_user_data))

        access_token = create_access_token(data={"sub": email})
        response = RedirectResponse(url=f"/?token={access_token}")
        return response
        
    except Exception as e:
        print(f"OAuth callback error: {e}")
        raise HTTPException(status_code=500, detail=f"Authentication failed: {str(e)}")

@app.get("/users/me", response_model=User, response_model_by_alias=False)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

@app.patch("/users/me", response_model=User, response_model_by_alias=False)
async def update_user_me(user_update: UserUpdate, current_user: User = Depends(get_current_user)):
    """Updates the current user's preferences, like default language."""
    update_data = user_update.model_dump(exclude_unset=True)
    if not update_data:
        return current_user
    
    updated_user = await users_collection.find_one_and_update(
        {"_id": ObjectId(current_user.id)},
        {"$set": update_data},
        return_document=True
    )
    updated_user['_id'] = str(updated_user['_id'])
    return User(**updated_user)

@app.get("/chats", response_model=List[ChatDetail], response_model_by_alias=False)
async def get_user_chats(current_user: User = Depends(get_current_user)):
    """Gets all chats for the current user with detailed participant and last message info."""
    chats_cursor = chats_collection.find({"participants": current_user.id})
    chat_details = []
    async for chat in chats_cursor:
        participant_ids = [ObjectId(p_id) for p_id in chat["participants"]]
        participants_cursor = users_collection.find({"_id": {"$in": participant_ids}})
        participants_list = []
        async for p in participants_cursor:
             participants_list.append(ChatParticipant(id=str(p["_id"]), name=p["name"], picture=p.get("picture")))
        
        last_message = await messages_collection.find_one(
            {"chat_id": str(chat["_id"])},
            sort=[("created_at", DESCENDING)]
        )
        
        detail = ChatDetail(
            _id=str(chat["_id"]),
            participants=participants_list,
            last_message_content=decrypt_message(last_message["original_content"]) if last_message else None,
            last_message_timestamp=last_message["created_at"] if last_message else None,
            created_at=chat["created_at"]
        )
        chat_details.append(detail)
    return chat_details

@app.post("/chats", response_model=ChatDetail, response_model_by_alias=False)
async def create_chat(participant_email: str, current_user: User = Depends(get_current_user)):
    """Creates a new chat with another user or returns the existing one."""
    if participant_email == current_user.email:
        raise HTTPException(status_code=400, detail="You cannot create a chat with yourself.")

    participant = await users_collection.find_one({"email": participant_email})
    if not participant:
        raise HTTPException(status_code=404, detail="User with that email not found.")
    
    participant_id_str = str(participant['_id'])
    
    existing_chat = await chats_collection.find_one({
        "participants": {"$all": [current_user.id, participant_id_str], "$size": 2}
    })

    if existing_chat:
        chat_id_str = str(existing_chat['_id'])
    else:
        new_chat_data = {"participants": [current_user.id, participant_id_str], "created_at": datetime.utcnow()}
        result = await chats_collection.insert_one(new_chat_data)
        chat_id_str = str(result.inserted_id)

    # Fetch the full chat detail to return
    chat_doc = await chats_collection.find_one({"_id": ObjectId(chat_id_str)})
    if not chat_doc:
        raise HTTPException(status_code=404, detail="Chat not found")
    p_ids = [ObjectId(p_id) for p_id in chat_doc["participants"]]
    p_cursor = users_collection.find({"_id": {"$in": p_ids}})
    p_list = [ChatParticipant(id=str(p["_id"]), name=p["name"], picture=p.get("picture")) async for p in p_cursor]
    
    return ChatDetail(_id=chat_id_str, participants=p_list, created_at=chat_doc["created_at"])

@app.get("/chats/{chat_id}/messages", response_model=List[Message], response_model_by_alias=False)
async def get_chat_messages(chat_id: str, current_user: User = Depends(get_current_user)):
    """Retrieves all messages for a specific chat, decrypting them for the client."""
    # Add validation for chat_id to prevent InvalidId errors
    if not chat_id or chat_id == 'undefined':
        raise HTTPException(status_code=400, detail="A valid chat ID must be provided.")
    try:
        chat_object_id = ObjectId(chat_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail=f"The provided chat ID '{chat_id}' is not valid.")

    chat = await chats_collection.find_one({"_id": chat_object_id, "participants": current_user.id})
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found or you are not a participant.")

    messages_data = []
    messages_cursor = messages_collection.find({"chat_id": chat_id}).sort("created_at", ASCENDING)
    async for msg in messages_cursor:
        try:
            msg_obj = {
                "_id": str(msg["_id"]), "chat_id": msg["chat_id"], "sender_id": msg["sender_id"],
                "original_content": decrypt_message(msg["original_content"]),
                "translations": {lang: decrypt_message(text) for lang, text in msg.get("translations", {}).items()},
                "created_at": msg["created_at"],
            }
            messages_data.append(Message(**msg_obj))
        except Exception as e:
            print(f"Skipping undecryptable message {msg['_id']}: {e}")
            continue
    return messages_data

# --- Debug & Maintenance Endpoints ---
@app.post("/auth/cleanup-states")
async def cleanup_oauth_states():
    """Manually cleanup expired OAuth states (for maintenance)."""
    initial_count = len(oauth_states)
    cleanup_expired_states()
    final_count = len(oauth_states)
    return {"message": f"Cleaned up {initial_count - final_count} expired states. {final_count} states remaining."}

@app.get("/debug/oauth-state")
async def debug_oauth_state():
    """Debug endpoint to check OAuth states."""
    cleanup_expired_states()  # Clean up before showing
    current_time = time.time()
    states_info = {}
    for state, timestamp in oauth_states.items():
        age_seconds = current_time - timestamp
        states_info[state[:10] + "..."] = {
            "age_seconds": age_seconds,
            "expired": age_seconds > 600
        }
    
    return {
        "active_states_count": len(oauth_states),
        "states_info": states_info
    }

@app.get("/debug/time")
async def debug_time():
    """Debug endpoint to check system time vs actual time."""
    import time
    system_timestamp = int(time.time())
    
    # Get actual time from a time server
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://worldtimeapi.org/api/timezone/Etc/UTC", timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                actual_timestamp = int(data['unixtime'])
                time_diff = system_timestamp - actual_timestamp
                
                return {
                    "system_time": system_timestamp,
                    "actual_time": actual_timestamp,
                    "difference_seconds": time_diff,
                    "system_readable": datetime.fromtimestamp(system_timestamp).isoformat(),
                    "actual_readable": datetime.fromtimestamp(actual_timestamp).isoformat(),
                    "needs_sync": abs(time_diff) > 5
                }
    except:
        pass
    
    return {
        "system_time": system_timestamp,
        "system_readable": datetime.fromtimestamp(system_timestamp).isoformat(),
        "note": "Could not fetch actual time for comparison"
    }


@app.get("/debug/cookies")
async def debug_cookies(request: Request):
    """Debug endpoint to check cookies and request details."""
    return {
        "cookies": dict(request.cookies),
        "query_params": dict(request.query_params),
        "headers": dict(request.headers),
        "url": str(request.url)
    }

# --- WebSocket Endpoint for Real-Time Chat ---
@app.websocket("/ws/{chat_id}")
async def websocket_endpoint(websocket: WebSocket, chat_id: str, token: str = Query(...)):
    """Handles the real-time WebSocket connection for a chat room."""
    # Add validation for chat_id to prevent crashes on connect
    if not chat_id or chat_id == 'undefined':
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    try:
        chat_object_id = ObjectId(chat_id)
    except InvalidId:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    try:
        current_user = await get_current_user(token)
        chat = await chats_collection.find_one({"_id": chat_object_id, "participants": current_user.id})
        if not chat:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
    except HTTPException:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await manager.connect(websocket, chat_id)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            content = message_data.get("content")

            if not content: continue

            encrypted_content = encrypt_message(content)
            new_message = {"chat_id": chat_id, "sender_id": current_user.id, "original_content": encrypted_content, "translations": {}, "created_at": datetime.utcnow()}
            result = await messages_collection.insert_one(new_message)
            message_id = result.inserted_id

            p_ids = [ObjectId(p_id) for p_id in chat["participants"]]
            p_cursor = users_collection.find({"_id": {"$in": p_ids}})
            target_languages = {p["default_language"] for p in await p_cursor.to_list(length=None)}
            
            translations = {}
            for lang in target_languages:
                if lang != "en": # Assuming 'en' is the base language of input
                    translated_text = await translate_text_gemini(content, lang)
                    translations[lang] = encrypt_message(translated_text)
            
            await messages_collection.update_one({"_id": message_id}, {"$set": {"translations": translations}})

            full_message = await messages_collection.find_one({"_id": message_id})
            if full_message:
                broadcast_message = {
                    "id": str(full_message["_id"]), 
                    "chat_id": full_message["chat_id"], 
                    "sender_id": full_message["sender_id"],
                    "original_content": content,
                    "translations": {lang: decrypt_message(text) for lang, text in translations.items()},
                    "created_at": full_message["created_at"].isoformat()
                }
                await manager.broadcast(json.dumps(broadcast_message), chat_id)
            else:
                print(f"Error: Could not retrieve full message for message ID {message_id}")

    except WebSocketDisconnect:
        manager.disconnect(websocket, chat_id)
    except Exception as e:
        print(f"WebSocket error in chat {chat_id}: {e}")
        manager.disconnect(websocket, chat_id)

# --- Main Execution ---
if __name__ == "__main__":
    # **IMPORTANT**: Create a 'client_secrets.json' file in the same directory.
    if not os.path.exists("client_secrets.json"):
        print("CRITICAL ERROR: 'client_secrets.json' not found.")
        print("Please download it from your Google Cloud Console and place it here.")
        print("\nRequired format:")
        print("""{
  "web": {
    "client_id": "your-client-id.apps.googleusercontent.com",
    "client_secret": "your-client-secret",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "redirect_uris": [
      "http://127.0.0.1:3000/auth/google/callback"
    ]
  }
}""")
    else:
        print("Starting LinguaLink server...")
        print("API will be available at http://127.0.0.1:3000")
        print("Interactive documentation at http://127.0.0.1:3000/docs")
        uvicorn.run("main:app", host="0.0.0.0", port=3000, reload=True)

