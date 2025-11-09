from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from datetime import datetime
from dotenv import load_dotenv
import os
import logging
from bson.objectid import ObjectId
from uuid import uuid4
from typing import Optional, List, Dict, Any

# Load environment variables
load_dotenv()
DB_URI = os.getenv("DB_URI")
DB_NAME = os.getenv("DB_NAME")

# Set up logging to app.log in the project root
logging.basicConfig(
    filename=os.path.join(os.path.dirname(__file__), "..", "..", "app.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def get_db_connection():
    """Establishes a connection to MongoDB and returns the database object."""
    try:
        client = MongoClient(DB_URI)
        client.admin.command("ping")  # Test the connection
        return client[DB_NAME]
    except ConnectionFailure as e:
        logging.error(f"Failed to connect to MongoDB: {str(e)}")
        raise

def initialize_database():
    """Ensures indexes exist for necessary collections in MongoDB."""
    db = get_db_connection()

    # application_logs
    db.application_logs.create_index("session_id")
    db.application_logs.create_index("created_at")
    logging.info("Indexes created for application_logs collection.")

    # document_store
    db.document_store.create_index("filename", unique=True)
    db.document_store.create_index("upload_timestamp")
    logging.info("Indexes created for document_store collection.")

    # sessions (metadata for chat list)
    db.sessions.create_index("updated_at")
    db.sessions.create_index([("title", "text"), ("last_message", "text")])
    logging.info("Indexes created for sessions collection.")
    logging.info("Database initialization complete.")

# ---------- Sessions (ChatGPT-like) ----------

def create_session(title: str = "New chat") -> str:
    """Create a new chat session and return its id (string)."""
    db = get_db_connection()
    sid = str(uuid4())
    now = datetime.utcnow()
    db.sessions.insert_one({
        "_id": sid,
        "title": title,
        "created_at": now,
        "updated_at": now,
        "last_message": ""
    })
    logging.info(f"Session created: {sid}")
    return sid

def get_sessions(limit: int = 50, query: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return recent sessions for sidebar, newest first."""
    db = get_db_connection()
    filt: Dict[str, Any] = {}
    if query:
        filt = {
            "$or": [
                {"title": {"$regex": query, "$options": "i"}},
                {"last_message": {"$regex": query, "$options": "i"}}
            ]
        }
    cur = db.sessions.find(filt).sort("updated_at", -1).limit(limit)
    return [
        {
            "id": doc["_id"],
            "title": doc.get("title", "Untitled"),
            "updated_at": doc.get("updated_at"),
            "last_message": doc.get("last_message", "")
        }
        for doc in cur
    ]

def rename_session(session_id: str, new_title: str) -> bool:
    db = get_db_connection()
    res = db.sessions.update_one(
        {"_id": session_id},
        {"$set": {"title": new_title, "updated_at": datetime.utcnow()}}
    )
    return res.modified_count == 1

def delete_session_and_logs(session_id: str) -> bool:
    """Delete a session metadata + all its logs."""
    db = get_db_connection()
    db.application_logs.delete_many({"session_id": session_id})
    res = db.sessions.delete_one({"_id": session_id})
    logging.info(f"Session delete {session_id}: deleted={res.deleted_count}")
    return res.deleted_count == 1

def touch_session(session_id: str, last_message: Optional[str] = None, title: Optional[str] = None) -> None:
    """Update updated_at and optionally last_message/title (for auto-title/preview)."""
    db = get_db_connection()
    update: Dict[str, Any] = {"updated_at": datetime.utcnow()}
    if last_message is not None:
        preview = (last_message or "").strip().replace("\n", " ")
        if len(preview) > 80:
            preview = preview[:77] + "..."
        update["last_message"] = preview
    if title is not None and title.strip():
        update["title"] = title.strip()
    db.sessions.update_one({"_id": session_id}, {"$set": update}, upsert=True)

# ---------- Logs ----------

def insert_application_logs(session_id: str, user_query: str, gpt_response: str, model: str) -> None:
    """Insert a log entry and keep session metadata fresh."""
    db = get_db_connection()
    db.application_logs.insert_one({
        "session_id": session_id,
        "user_query": user_query,
        "gpt_response": gpt_response,
        "model": model,
        "created_at": datetime.utcnow()
    })
    try:
        touch_session(session_id, last_message=user_query)
    except Exception as e:
        logging.warning(f"touch_session failed: {e}")
    logging.info("Log inserted successfully.")

def get_chat_history(session_id: str) -> List[Dict[str, str]]:
    """Return chat history in LangChain-style message dicts."""
    db = get_db_connection()
    logs = db.application_logs.find({"session_id": session_id}).sort("created_at", 1)
    messages: List[Dict[str, str]] = []
    for log in logs:
        messages.extend([
            {"role": "human", "content": log["user_query"]},
            {"role": "ai", "content": log["gpt_response"]}
        ])
    return messages

# ---------- Documents ----------

def insert_document_record(filename: str) -> str:
    """Insert a document record into document_store."""
    db = get_db_connection()
    document = {
        "filename": filename,
        "upload_timestamp": datetime.utcnow()
    }
    result = db.document_store.insert_one(document)
    return str(result.inserted_id)

def delete_document_record(file_id: str) -> bool:
    """Delete a document record by ObjectId string."""
    db = get_db_connection()
    try:
        obj_id = ObjectId(file_id)
    except Exception as e:
        logging.error(f"Invalid file_id: {file_id}, error: {str(e)}")
        return False
    result = db.document_store.delete_one({"_id": obj_id})
    logging.info(f"Delete result for file_id {file_id}: deleted_count={result.deleted_count}")
    return result.deleted_count > 0

def get_all_documents():
    """List documents sorted by latest uploads."""
    db = get_db_connection()
    documents = db.document_store.find().sort("upload_timestamp", -1)
    return [
        {
            "id": str(doc["_id"]),
            "filename": doc["filename"],
            "upload_timestamp": doc["upload_timestamp"]
        }
        for doc in documents
    ]
