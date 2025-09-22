# main.py
#
# This file contains the complete backend for LinguaLink, a real-time chat application
# built with FastAPI, MongoDB, and integrated with the Gemini API for translation.
#
# Version 1.7.0 (Consolidated Foundation)
# - SECURITY: Secrets loaded from environment variables.
# - SECURITY: Server-side validation for file uploads.
# - PERFORMANCE: Database indexes for users, chats, and messages.
# - FEATURE: Real-time "user is typing..." indicators.
# - FEATURE: Message Status Indicators (Sent, Delivered, Read).

import os
import json
import asyncio
import secrets
import time
import shutil
from datetime import datetime
from typing import List, Dict, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, Depends, HTTPException, status, WebSocketDisconnect, Query, Request, UploadFile, File
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from bson.errors import InvalidId
from jose import jwt, JWTError
from cryptography.fernet import Fernet, InvalidToken
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from google_auth_oauthlib.flow import Flow
from pymongo import ASCENDING, DESCENDING, IndexModel, TEXT
from dotenv import load_dotenv

load_dotenv() 

# --- Configuration ---
try:
    MONGO_DETAILS = os.environ["MONGO_DETAILS"]
    ENCRYPTION_KEY = os.environ["ENCRYPTION_KEY"]
    GOOGLE_CLIENT_ID = os.environ["GOOGLE_CLIENT_ID"]
    GOOGLE_CLIENT_SECRET = os.environ["GOOGLE_CLIENT_SECRET"]
    GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
    JWT_SECRET_KEY = os.environ["JWT_SECRET_KEY"]
    REDIRECT_URI = os.environ.get("REDIRECT_URI", "http://127.0.0.1:3000/auth/google/callback")
except KeyError as e:
    raise RuntimeError(f"FATAL: Environment variable {e} is not set. Application cannot start.") from e

fernet = Fernet(ENCRYPTION_KEY.encode())
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
ALGORITHM = "HS256"
ALLOWED_UPLOAD_CONTENT_TYPES = [ "image/jpeg", "image/png", "image/gif", "image/webp", "audio/mpeg", "audio/ogg", "audio/wav", "application/pdf", "text/plain", "application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-powerpoint", "application/vnd.openxmlformats-officedocument.presentationml.presentation" ]
oauth_states: Dict[str, float] = {}

# --- Initialization ---
app = FastAPI(
    title="LinguaLink API",
    description="Backend for a WhatsApp-style chat application with real-time translation and presence.",
    version="1.7.0"
)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/files", StaticFiles(directory=UPLOAD_DIR), name="files")
app.add_middleware( CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],)

# --- Database Connection ---
client = AsyncIOMotorClient(MONGO_DETAILS)
db = client.lingualink_db
users_collection = db.get_collection("users")
chats_collection = db.get_collection("chats")
messages_collection = db.get_collection("messages")

# --- Database Indexing ---
@app.on_event("startup")
async def create_db_indexes():
    await users_collection.create_indexes([IndexModel([("email", ASCENDING)], unique=True)])
    await chats_collection.create_indexes([IndexModel([("participants", ASCENDING)])])
    await messages_collection.create_indexes([
        IndexModel([("chat_id", ASCENDING), ("created_at", DESCENDING)]),
        IndexModel([("status", ASCENDING), ("sender_id", ASCENDING)]),
        IndexModel([("original_content", TEXT)])
    ])
    print("Database indexes have been created/verified.")

# --- Pydantic Models ---
class User(BaseModel):
    id: str = Field(..., alias="_id")
    google_id: Optional[str] = None; email: str; name: str
    picture: Optional[str] = None; default_language: str = "en"
    is_online: bool = False; last_seen: Optional[datetime] = None

class UserUpdate(BaseModel):
    default_language: Optional[str] = None

class Message(BaseModel):
    id: str = Field(..., alias="_id")
    chat_id: str; sender_id: str
    message_type: str = "text"
    original_content: str
    file_url: Optional[str] = None
    translations: Dict[str, str] = {}
    reactions: Dict[str, List[str]] = Field(default_factory=dict)
    created_at: datetime
    status: str = "sent"  # sent, delivered, read
    read_by: List[str] = Field(default_factory=list)


class ChatParticipant(BaseModel):
    id: str; name: str; picture: Optional[str] = None
    is_online: bool; last_seen: Optional[datetime] = None

class ChatDetail(BaseModel):
    id: str = Field(..., alias="_id")
    participants: List[ChatParticipant]
    last_message_content: Optional[str] = None
    last_message_timestamp: Optional[datetime] = None
    last_message_type: str = "text"
    created_at: datetime
    unread_count: int = 0

# --- WebSocket Connection Manager ---
class ConnectionManager:
    def __init__(self):
        self.user_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept(); self.user_connections[user_id] = websocket

    def disconnect(self, user_id: str):
        if user_id in self.user_connections: del self.user_connections[user_id]

    async def broadcast_to_users(self, user_ids: List[str], message: str):
        for user_id in user_ids:
            if user_id in self.user_connections:
                await self.user_connections[user_id].send_text(message)

manager = ConnectionManager()

# --- Utility Functions ---
def create_access_token(data: dict):
    return jwt.encode(data, JWT_SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Query(...)):
    credentials_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub", "")
        if not email: raise credentials_exception
    except JWTError: raise credentials_exception
    user = await users_collection.find_one({"email": email})
    if user is None: raise credentials_exception
    user['_id'] = str(user['_id'])
    return User(**user)

def encrypt_message(text: str) -> str: return fernet.encrypt(text.encode()).decode()

def decrypt_message(encrypted_text: str) -> str:
    try: return fernet.decrypt(encrypted_text.encode()).decode()
    except (InvalidToken, TypeError): return "[Message could not be decrypted]"

async def translate_text_gemini(text: str, target_language: str) -> str:
    if not GEMINI_API_KEY or not text: return text
    headers = {"Content-Type": "application/json"}
    prompt = f"Identify the language of the following text, which might be in English letters but represent another language (like Hinglish or Tanglish). Then, translate it to {target_language}. Provide only the final translation. Text: '{text}'"
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

# --- Presence & Message Status Logic ---
async def update_and_broadcast_presence(user_id: str, is_online: bool):
    await users_collection.update_one({"_id": ObjectId(user_id)}, {"$set": {"is_online": is_online, "last_seen": datetime.utcnow() if not is_online else None}})
    
    user_chats_cursor = chats_collection.find({"participants": user_id})
    contact_ids = set()
    async for chat in user_chats_cursor:
        for p_id in chat['participants']:
            if p_id != user_id: contact_ids.add(p_id)
    
    presence_message = json.dumps({ "type": "presence_update", "user_id": user_id, "is_online": is_online, "last_seen": datetime.utcnow().isoformat() if not is_online else None })
    await manager.broadcast_to_users(list(contact_ids), presence_message)
    
    if is_online:
        await confirm_delivery_for_user(user_id)

async def confirm_delivery_for_user(user_id: str):
    undelivered_cursor = messages_collection.find({
        "status": "sent",
        "chat_id": {"$in": [str(c["_id"]) async for c in chats_collection.find({"participants": user_id})]},
        "sender_id": {"$ne": user_id}
    })
    async for msg in undelivered_cursor:
        await messages_collection.update_one({"_id": msg["_id"]}, {"$set": {"status": "delivered"}})
        chat = await chats_collection.find_one({"_id": ObjectId(msg["chat_id"])})
        if chat:
            delivery_update = json.dumps({"type": "message_delivered", "id": str(msg["_id"]), "chat_id": msg["chat_id"]})
            await manager.broadcast_to_users(chat["participants"], delivery_update)

# --- Background Task for Translation ---
async def process_translations(message_id: ObjectId, chat_id: str, sender: User, original_content: str):
    chat = await chats_collection.find_one({"_id": ObjectId(chat_id)})
    if not chat: return
    other_participant_ids = [ObjectId(p_id) for p_id in chat["participants"] if p_id != sender.id]
    p_cursor = users_collection.find({"_id": {"$in": other_participant_ids}})
    translations = {sender.default_language: encrypt_message(original_content)}
    async for participant in p_cursor:
        lang = participant.get("default_language", "en")
        if lang != sender.default_language:
            translated_text = await translate_text_gemini(original_content, lang)
            translations[lang] = encrypt_message(translated_text)
    await messages_collection.update_one({"_id": message_id}, {"$set": {"translations": translations}})
    decrypted_translations = {lang: decrypt_message(text) for lang, text in translations.items()}
    broadcast_msg = json.dumps({ "type": "translations_ready", "id": str(message_id), "chat_id": chat_id, "translations": decrypted_translations })
    await manager.broadcast_to_users(chat["participants"], broadcast_msg)

# --- API Endpoints ---
@app.post("/uploads")
async def upload_file(file: UploadFile = File(...), token: str = Query(...)):
    await get_current_user(token)
    if file.content_type not in ALLOWED_UPLOAD_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"File type '{file.content_type}' not allowed.")
    file_extension = os.path.splitext(file.filename)[1]
    safe_filename = f"{secrets.token_hex(16)}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    try:
        with open(file_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
    finally: file.file.close()
    return {"file_url": f"/files/{safe_filename}"}

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
        p_list = [ ChatParticipant(id=str(p["_id"]), name=p["name"], picture=p.get("picture"), is_online=p.get("is_online", False), last_seen=p.get("last_seen")) async for p in p_cursor ]
        last_message = await messages_collection.find_one({"chat_id": str(chat["_id"])}, sort=[("created_at", DESCENDING)])
        last_message_content = None; last_message_type = "text"
        if last_message:
            last_message_type = last_message.get("message_type", "text")
            if last_message_type == "text": last_message_content = decrypt_message(last_message["original_content"])
            else: last_message_content = last_message["original_content"]
        
        unread_count = await messages_collection.count_documents({
            "chat_id": str(chat["_id"]),
            "sender_id": {"$ne": current_user.id},
            "read_by": {"$nin": [current_user.id]}
        })
        
        detail = ChatDetail(_id=str(chat["_id"]), participants=p_list, last_message_content=last_message_content, last_message_timestamp=last_message.get("created_at") if last_message else None, last_message_type=last_message_type, created_at=chat["created_at"], unread_count=unread_count)
        chat_details.append(detail)
    return chat_details

@app.post("/chats", response_model=ChatDetail, response_model_by_alias=False)
async def create_chat(participant_email: str, current_user: User = Depends(get_current_user)):
    if participant_email == current_user.email: raise HTTPException(status_code=400, detail="You cannot create a chat with yourself.")
    participant = await users_collection.find_one({"email": participant_email})
    if not participant: raise HTTPException(status_code=404, detail="User with that email not found.")
    participant_id_str = str(participant['_id'])
    existing_chat = await chats_collection.find_one({"participants": {"$all": [current_user.id, participant_id_str], "$size": 2}})
    if existing_chat: chat_id_str = str(existing_chat['_id'])
    else:
        new_chat_data = {"participants": [current_user.id, participant_id_str], "created_at": datetime.utcnow()}
        result = await chats_collection.insert_one(new_chat_data)
        chat_id_str = str(result.inserted_id)
    chat_doc = await chats_collection.find_one({"_id": ObjectId(chat_id_str)})
    p_ids = [ObjectId(p_id) for p_id in chat_doc["participants"]]
    p_cursor = users_collection.find({"_id": {"$in": p_ids}})
    p_list = [ ChatParticipant(id=str(p["_id"]), name=p["name"], picture=p.get("picture"), is_online=p.get("is_online", False), last_seen=p.get("last_seen")) async for p in p_cursor ]
    return ChatDetail(_id=chat_id_str, participants=p_list, created_at=chat_doc["created_at"])

@app.get("/chats/{chat_id}/messages", response_model=List[Message], response_model_by_alias=False)
async def get_chat_messages(chat_id: str, current_user: User = Depends(get_current_user)):
    try: chat_object_id = ObjectId(chat_id)
    except InvalidId: raise HTTPException(status_code=400, detail="Invalid chat ID.")
    chat = await chats_collection.find_one({"_id": chat_object_id, "participants": current_user.id})
    if not chat: raise HTTPException(status_code=404, detail="Chat not found.")
    messages_data = []
    messages_cursor = messages_collection.find({"chat_id": chat_id}).sort("created_at", ASCENDING)
    user_lang = current_user.default_language
    async for msg in messages_cursor:
        if msg.get("message_type", "text") == "text":
            decrypted_original = decrypt_message(msg["original_content"])
            msg["original_content"] = decrypted_original
            decrypted_translations = {lang: decrypt_message(text) for lang, text in msg.get("translations", {}).items()}
            if user_lang not in decrypted_translations:
                translated_text = await translate_text_gemini(decrypted_original, user_lang)
                if translated_text != decrypted_original:
                    encrypted_new_translation = encrypt_message(translated_text)
                    await messages_collection.update_one( {"_id": msg["_id"]}, {"$set": {f"translations.{user_lang}": encrypted_new_translation}})
                    decrypted_translations[user_lang] = translated_text
            msg["translations"] = decrypted_translations
        msg["_id"] = str(msg["_id"])
        messages_data.append(Message(**msg))
    return messages_data

@app.post("/chats/{chat_id}/read")
async def mark_messages_as_read(chat_id: str, current_user: User = Depends(get_current_user)):
    chat = await chats_collection.find_one({"_id": ObjectId(chat_id), "participants": current_user.id})
    if not chat: raise HTTPException(status_code=404, detail="Chat not found.")
    
    result = await messages_collection.update_many(
        {"chat_id": chat_id, "sender_id": {"$ne": current_user.id}, "read_by": {"$nin": [current_user.id]}},
        {"$addToSet": {"read_by": current_user.id}, "$set": {"status": "read"}}
    )
    if result.modified_count > 0:
        read_update = json.dumps({"type": "messages_read", "chat_id": chat_id, "reader_id": current_user.id})
        await manager.broadcast_to_users(chat["participants"], read_update)
    return {"status": "success", "modified_count": result.modified_count}


# --- WebSocket Endpoint ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = Query(...)):
    try: current_user = await get_current_user(token)
    except HTTPException: await websocket.close(code=status.WS_1008_POLICY_VIOLATION); return
    await manager.connect(websocket, current_user.id)
    await update_and_broadcast_presence(current_user.id, is_online=True)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data); msg_type = message_data.get("type"); chat_id = message_data.get("chat_id")
            if not chat_id: continue
            chat = await chats_collection.find_one({"_id": ObjectId(chat_id), "participants": current_user.id})
            if not chat: continue
            other_participants = [p for p in chat["participants"] if p != current_user.id]
            if msg_type == "message":
                message_content_type = message_data.get("message_type", "text")
                new_message = { "chat_id": chat_id, "sender_id": current_user.id, "message_type": message_content_type, "created_at": datetime.utcnow(), "translations": {}, "reactions": {}, "status": "sent", "read_by": []}
                content_to_broadcast = ""
                if message_content_type == "text":
                    content = message_data.get("content"); new_message["original_content"] = encrypt_message(content); content_to_broadcast = content
                else: 
                    file_url, original_filename = message_data.get("file_url"), message_data.get("original_content")
                    new_message["file_url"], new_message["original_content"] = file_url, original_filename; content_to_broadcast = original_filename
                
                is_recipient_online = any(p_id in manager.user_connections for p_id in other_participants)
                if is_recipient_online: new_message["status"] = "delivered"
                
                result = await messages_collection.insert_one(new_message)
                message_id = result.inserted_id
                
                instant_broadcast_msg = { "type": "new_message", "id": str(message_id), "chat_id": chat_id, "sender_id": current_user.id, "sender_picture": current_user.picture, "message_type": message_content_type, "original_content": content_to_broadcast, "file_url": new_message.get("file_url"), "translations": {}, "reactions": {}, "created_at": new_message["created_at"].isoformat(), "status": new_message["status"], "read_by": [] }
                await manager.broadcast_to_users(chat["participants"], json.dumps(instant_broadcast_msg))

                if message_content_type == "text": asyncio.create_task(process_translations(message_id, chat_id, current_user, content_to_broadcast))

            elif msg_type == "reaction":
                message_id_str = message_data.get("message_id"); emoji = message_data.get("emoji")
                message_id = ObjectId(message_id_str); message = await messages_collection.find_one({"_id": message_id})
                if not message: continue
                reactions = message.get("reactions", {}); user_id = current_user.id
                if emoji not in reactions: reactions[emoji] = []
                if user_id in reactions[emoji]: reactions[emoji].remove(user_id)
                else: reactions[emoji].append(user_id)
                if not reactions[emoji]: del reactions[emoji] 
                await messages_collection.update_one({"_id": message_id}, {"$set": {"reactions": reactions}})
                reaction_update_msg = json.dumps({ "type": "reaction_update", "chat_id": chat_id, "message_id": message_id_str, "reactions": reactions })
                await manager.broadcast_to_users(chat["participants"], reaction_update_msg)

            elif msg_type == "typing":
                typing_status_msg = json.dumps({ "type": "typing_status", "chat_id": chat_id, "user_id": current_user.id, "user_name": current_user.name, "is_typing": message_data.get("is_typing", False) })
                await manager.broadcast_to_users(other_participants, typing_status_msg)

    except WebSocketDisconnect: pass
    except Exception as e: print(f"WebSocket error for user {current_user.id}: {e}")
    finally:
        manager.disconnect(current_user.id)
        await update_and_broadcast_presence(current_user.id, is_online=False)

# --- OAuth Routes ---
@app.get("/auth/google/signin")
async def auth_google_signin():
    flow = Flow.from_client_secrets_file( client_secrets_file="client_secrets.json", scopes=['openid', 'https://www.googleapis.com/auth/userinfo.email', 'https://www.googleapis.com/auth/userinfo.profile'], redirect_uri=REDIRECT_URI )
    authorization_url, state = flow.authorization_url(access_type='offline', include_granted_scopes='true')
    oauth_states[state] = time.time()
    return RedirectResponse(authorization_url)

@app.get("/auth/google/callback")
async def auth_google_callback(request: Request):
    state = request.query_params.get("state")
    if not state or state not in oauth_states or time.time() - oauth_states[state] > 600: raise HTTPException(status_code=401, detail="Invalid or expired state.")
    del oauth_states[state]
    flow = Flow.from_client_secrets_file( client_secrets_file="client_secrets.json", scopes=None, state=state, redirect_uri=REDIRECT_URI )
    flow.fetch_token(authorization_response=str(request.url))
    credentials = flow.credentials
    try:
        id_info = id_token.verify_oauth2_token( credentials.id_token, google_requests.Request(), GOOGLE_CLIENT_ID, clock_skew_in_seconds=15 )
    except ValueError as e: raise HTTPException(status_code=401, detail=f"Token verification failed: {e}")
    email = id_info['email']
    user = await users_collection.find_one({"email": email})
    if not user:
        new_user_data = { "google_id": id_info['sub'], "email": email, "name": id_info.get("name"), "picture": id_info.get("picture"), "default_language": "en", "created_at": datetime.utcnow(), "is_online": False }
        await users_collection.insert_one(new_user_data)
    access_token = create_access_token(data={"sub": email})
    return RedirectResponse(url=f"/?token={access_token}")

if __name__ == "__main__":
    print("--- LINGUALINK SERVER STARTING ---")
    print("Reminder: Ensure all required environment variables are set.")
    uvicorn.run("main:app", host="0.0.0.0", port=3000, reload=True)