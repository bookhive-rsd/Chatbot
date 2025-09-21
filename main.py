# main.py
#
# This file contains the complete backend for LinguaLink, a real-time chat application
# built with FastAPI, MongoDB, and integrated with the Gemini API for translation.
#
# Version 1.4.4 Features:
# - Added robust error handling for `InvalidToken` to prevent crashes from old data.
# - Fixed `InvalidToken` decryption error by using a persistent encryption key.
# - Fixed "Token used too early" error by adding clock skew tolerance to OAuth.
# - Corrected JWT algorithm from HS26 to HS256.
# - Real-time presence system (online/offline status and last seen timestamp).
# - Instant message delivery with translations handled as a background task.
# - WebSocket broadcasting for presence updates to relevant users.

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
from cryptography.fernet import Fernet, InvalidToken
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from google_auth_oauthlib.flow import Flow
import httpx
from pymongo import ASCENDING, DESCENDING

# --- Configuration ---
MONGO_DETAILS = os.environ.get("MONGO_DETAILS", "mongodb://localhost:27017")

# IMPORTANT: This key must be persistent. For production, load it from an environment
# variable. Using a fixed key here for development with auto-reloading.
# You can generate a new one with:
# python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
FIXED_DEV_ENCRYPTION_KEY = "v2qL4z-4x_gE8sB_nMcRfUjXn2r5u8xAzDCF-JaNdSg="
ENCRYPTION_KEY = os.environ.get("ENCRYPTION_KEY", FIXED_DEV_ENCRYPTION_KEY)
fernet = Fernet(ENCRYPTION_KEY.encode())

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "660588293356-insq0j2214him7fq448sskegc7frnmhf.apps.googleusercontent.com")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "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")
REDIRECT_URI = "http://127.0.0.1:3000/auth/google/callback"
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBcnPIGkKdkSpoJaPv3W3mw3uV7c9pH2QI")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "a_very_secret_key_for_jwt_tokens")
ALGORITHM = "HS256"

oauth_states: Dict[str, float] = {}

# --- Initialization ---
app = FastAPI(
    title="LinguaLink API",
    description="Backend for a WhatsApp-style chat application with real-time translation and presence.",
    version="1.4.4"
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- Database Connection ---
client = AsyncIOMotorClient(MONGO_DETAILS)
db = client.lingualink_db
users_collection = db.get_collection("users")
chats_collection = db.get_collection("chats")
messages_collection = db.get_collection("messages")

# --- Pydantic Models ---
class User(BaseModel):
    id: str = Field(..., alias="_id")
    google_id: Optional[str] = None
    email: str
    name: str
    picture: Optional[str] = None
    default_language: str = "en"
    is_online: bool = False
    last_seen: Optional[datetime] = None

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
    is_online: bool
    last_seen: Optional[datetime] = None

class ChatDetail(BaseModel):
    id: str = Field(..., alias="_id")
    participants: List[ChatParticipant]
    last_message_content: Optional[str] = None
    last_message_timestamp: Optional[datetime] = None
    created_at: datetime

# --- WebSocket Connection Manager ---
class ConnectionManager:
    def __init__(self):
        self.user_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.user_connections[user_id] = websocket

    def disconnect(self, user_id: str):
        if user_id in self.user_connections:
            del self.user_connections[user_id]

    async def broadcast_to_users(self, user_ids: List[str], message: str):
        for user_id in user_ids:
            if user_id in self.user_connections:
                await self.user_connections[user_id].send_text(message)

manager = ConnectionManager()

# --- Utility Functions (Auth, Crypto, etc.) ---
def create_access_token(data: dict):
    return jwt.encode(data, JWT_SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Query(...)):
    credentials_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub", "")
        if not email: raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = await users_collection.find_one({"email": email})
    if user is None: raise credentials_exception
    user['_id'] = str(user['_id'])
    return User(**user)

def encrypt_message(text: str) -> str: return fernet.encrypt(text.encode()).decode()

def decrypt_message(encrypted_text: str) -> str:
    try:
        return fernet.decrypt(encrypted_text.encode()).decode()
    except (InvalidToken, TypeError):
        # This can happen if the encryption key changes or the data is corrupt
        print(f"Warning: Could not decrypt a message. Returning placeholder text.")
        return "[Message could not be decrypted]"

async def translate_text_gemini(text: str, target_language: str) -> str:
    if not GEMINI_API_KEY or not text: return text
    headers = {"Content-Type": "application/json"}
    prompt = f"Translate the following text to {target_language}. Provide only the translation, without any preamble or explanation: '{text}'"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(GEMINI_API_URL, json=payload, headers=headers, timeout=10.0)
            response.raise_for_status()
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text'].strip()
        except Exception as e:
            print(f"Gemini translation error: {e}")
            return text

# --- Presence Management ---
async def update_and_broadcast_presence(user_id: str, is_online: bool):
    """Updates user presence in DB and broadcasts to their contacts."""
    update_data = {"is_online": is_online}
    if not is_online:
        update_data["last_seen"] = datetime.utcnow()
    
    await users_collection.update_one({"_id": ObjectId(user_id)}, {"$set": update_data})
    
    user_chats_cursor = chats_collection.find({"participants": user_id})
    contact_ids = set()
    async for chat in user_chats_cursor:
        for p_id in chat['participants']:
            if p_id != user_id:
                contact_ids.add(p_id)
    
    last_seen_value = update_data.get("last_seen")

    presence_message = json.dumps({
        "type": "presence_update",
        "user_id": user_id,
        "is_online": is_online,
        "last_seen": last_seen_value.isoformat() if last_seen_value else None
    })
    await manager.broadcast_to_users(list(contact_ids), presence_message)

# --- Background Task for Translation ---
async def process_translations(message_id: ObjectId, chat_id: str, sender: User, original_content: str):
    """Fetches translations and updates the message, then broadcasts the result."""
    chat = await chats_collection.find_one({"_id": ObjectId(chat_id)})
    if not chat: return

    other_participant_ids = [ObjectId(p_id) for p_id in chat["participants"] if p_id != sender.id]
    p_cursor = users_collection.find({"_id": {"$in": other_participant_ids}})
    
    translations = {}
    translations[sender.default_language] = encrypt_message(original_content)

    async for participant in p_cursor:
        lang = participant.get("default_language", "en")
        if lang != sender.default_language:
            translated_text = await translate_text_gemini(original_content, lang)
            translations[lang] = encrypt_message(translated_text)

    await messages_collection.update_one({"_id": message_id}, {"$set": {"translations": translations}})
    
    decrypted_translations = {lang: decrypt_message(text) for lang, text in translations.items()}
    
    broadcast_msg = json.dumps({
        "type": "translations_ready",
        "id": str(message_id),
        "chat_id": chat_id, # Added for context on the client
        "translations": decrypted_translations
    })
    
    await manager.broadcast_to_users(chat["participants"], broadcast_msg)

# --- API Endpoints ---
@app.get("/", response_class=FileResponse)
async def read_root(): return FileResponse("index.html")

@app.get("/users/me", response_model=User, response_model_by_alias=False)
async def read_users_me(current_user: User = Depends(get_current_user)): return current_user

@app.patch("/users/me", response_model=User, response_model_by_alias=False)
async def update_user_me(user_update: UserUpdate, current_user: User = Depends(get_current_user)):
    update_data = user_update.model_dump(exclude_unset=True)
    if not update_data: return current_user
    
    updated_user = await users_collection.find_one_and_update(
        {"_id": ObjectId(current_user.id)}, {"$set": update_data}, return_document=True
    )
    updated_user['_id'] = str(updated_user['_id'])
    return User(**updated_user)

@app.get("/chats", response_model=List[ChatDetail], response_model_by_alias=False)
async def get_user_chats(current_user: User = Depends(get_current_user)):
    chats_cursor = chats_collection.find({"participants": current_user.id})
    chat_details = []
    async for chat in chats_cursor:
        participant_ids = [ObjectId(p_id) for p_id in chat["participants"]]
        p_cursor = users_collection.find({"_id": {"$in": participant_ids}})
        p_list = [
            ChatParticipant(
                id=str(p["_id"]), name=p["name"], picture=p.get("picture"),
                is_online=p.get("is_online", False), last_seen=p.get("last_seen")
            ) async for p in p_cursor
        ]
        
        last_message = await messages_collection.find_one(
            {"chat_id": str(chat["_id"])}, sort=[("created_at", DESCENDING)]
        )
        
        # Safely decrypt last message content
        last_message_content = None
        if last_message and "original_content" in last_message:
            last_message_content = decrypt_message(last_message["original_content"])

        detail = ChatDetail(
            _id=str(chat["_id"]), participants=p_list,
            last_message_content=last_message_content,
            last_message_timestamp=last_message["created_at"] if last_message else None,
            created_at=chat["created_at"]
        )
        chat_details.append(detail)
    return chat_details

@app.post("/chats", response_model=ChatDetail, response_model_by_alias=False)
async def create_chat(participant_email: str, current_user: User = Depends(get_current_user)):
    if participant_email == current_user.email:
        raise HTTPException(status_code=400, detail="You cannot create a chat with yourself.")

    participant = await users_collection.find_one({"email": participant_email})
    if not participant:
        raise HTTPException(status_code=404, detail="User with that email not found.")
    
    participant_id_str = str(participant['_id'])
    
    existing_chat = await chats_collection.find_one(
        {"participants": {"$all": [current_user.id, participant_id_str], "$size": 2}}
    )

    if existing_chat:
        chat_id_str = str(existing_chat['_id'])
    else:
        new_chat_data = {"participants": [current_user.id, participant_id_str], "created_at": datetime.utcnow()}
        result = await chats_collection.insert_one(new_chat_data)
        chat_id_str = str(result.inserted_id)

    chat_doc = await chats_collection.find_one({"_id": ObjectId(chat_id_str)})
    p_ids = [ObjectId(p_id) for p_id in chat_doc["participants"]]
    p_cursor = users_collection.find({"_id": {"$in": p_ids}})
    p_list = [
        ChatParticipant(
            id=str(p["_id"]), name=p["name"], picture=p.get("picture"),
            is_online=p.get("is_online", False), last_seen=p.get("last_seen")
        ) async for p in p_cursor
    ]
    
    return ChatDetail(_id=chat_id_str, participants=p_list, created_at=chat_doc["created_at"])

@app.get("/chats/{chat_id}/messages", response_model=List[Message], response_model_by_alias=False)
async def get_chat_messages(chat_id: str, current_user: User = Depends(get_current_user)):
    try:
        chat_object_id = ObjectId(chat_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail=f"Invalid chat ID.")

    chat = await chats_collection.find_one({"_id": chat_object_id, "participants": current_user.id})
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found.")

    messages_data = []
    messages_cursor = messages_collection.find({"chat_id": chat_id}).sort("created_at", ASCENDING)
    async for msg in messages_cursor:
        msg["_id"] = str(msg["_id"])
        msg["original_content"]=decrypt_message(msg["original_content"])
        msg["translations"]={lang: decrypt_message(text) for lang, text in msg.get("translations", {}).items()}
        messages_data.append(Message(**msg))
            
    return messages_data

# --- WebSocket Endpoint ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = Query(...)):
    try:
        current_user = await get_current_user(token)
    except HTTPException:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await manager.connect(websocket, current_user.id)
    await update_and_broadcast_presence(current_user.id, is_online=True)

    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            chat_id = message_data.get("chat_id")
            content = message_data.get("content")
            if not chat_id or not content: continue
            
            chat = await chats_collection.find_one({"_id": ObjectId(chat_id), "participants": current_user.id})
            if not chat: continue

            encrypted_content = encrypt_message(content)
            new_message = {
                "chat_id": chat_id, "sender_id": current_user.id, 
                "original_content": encrypted_content, 
                "translations": {}, "created_at": datetime.utcnow()
            }
            result = await messages_collection.insert_one(new_message)
            message_id = result.inserted_id

            instant_broadcast_msg = json.dumps({
                "type": "new_message",
                "id": str(message_id),
                "chat_id": chat_id, "sender_id": current_user.id,
                "original_content": content,
                "translations": {},
                "created_at": new_message["created_at"].isoformat()
            })
            
            await manager.broadcast_to_users(chat["participants"], instant_broadcast_msg)

            asyncio.create_task(process_translations(
                message_id, chat_id, current_user, content
            ))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error for user {current_user.id}: {e}")
    finally:
        manager.disconnect(current_user.id)
        await update_and_broadcast_presence(current_user.id, is_online=False)

# --- OAuth Routes ---
@app.get("/auth/google/signin")
async def auth_google_signin():
    flow = Flow.from_client_secrets_file(
            client_secrets_file="client_secrets.json",
            scopes=['openid', 'https://www.googleapis.com/auth/userinfo.email', 'https://www.googleapis.com/auth/userinfo.profile'],
            redirect_uri=REDIRECT_URI
        )
    authorization_url, state = flow.authorization_url(access_type='offline', include_granted_scopes='true')
    oauth_states[state] = time.time()
    return RedirectResponse(authorization_url)

@app.get("/auth/google/callback")
async def auth_google_callback(request: Request):
    state = request.query_params.get("state")
    if not state or state not in oauth_states or time.time() - oauth_states[state] > 600:
        raise HTTPException(status_code=401, detail="Invalid or expired state.")
    del oauth_states[state]

    flow = Flow.from_client_secrets_file(
        client_secrets_file="client_secrets.json", scopes=None, state=state, redirect_uri=REDIRECT_URI
    )
    flow.fetch_token(authorization_response=str(request.url))
    credentials = flow.credentials
    
    try:
        id_info = id_token.verify_oauth2_token(
            credentials.id_token, 
            google_requests.Request(), 
            GOOGLE_CLIENT_ID,
            clock_skew_in_seconds=15
        )
    except ValueError as e:
        raise HTTPException(status_code=401, detail=f"Token verification failed: {e}")

    
    email = id_info['email']
    user = await users_collection.find_one({"email": email})
    if not user:
        new_user_data = {
            "google_id": id_info['sub'], "email": email, "name": id_info.get("name"), 
            "picture": id_info.get("picture"), "default_language": "en", 
            "created_at": datetime.utcnow(), "is_online": False
        }
        await users_collection.insert_one(new_user_data)
    
    access_token = create_access_token(data={"sub": email})
    return RedirectResponse(url=f"/?token={access_token}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=3000, reload=True)

