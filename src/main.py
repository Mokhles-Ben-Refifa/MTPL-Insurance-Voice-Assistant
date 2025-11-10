# src/main.py
import os
from dotenv import load_dotenv, find_dotenv

# --- Load env BEFORE any imports that touch LangChain/Chroma ---
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")  # silence Chroma noise
load_dotenv(find_dotenv(), override=True)

import uuid
import logging
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from functools import lru_cache

from src.pydantic_models import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest
from src.db_utils import (
    insert_application_logs, get_chat_history, get_all_documents,
    insert_document_record, delete_document_record
)
from src.chroma_utils import index_document_to_chroma, delete_doc_from_chroma, vectorstore

# Optional: check BM25 availability for /whoami
try:
    import rank_bm25  # noqa: F401
    _bm25_available = True
except Exception:
    _bm25_available = False

# Logging
logging.basicConfig(filename="app.log", level=logging.INFO)
logger = logging.getLogger(__name__)

# Import AFTER env is loaded so retriever builds with the right mode (dense/bm25/hybrid)
from src.langchain_utils import get_rag_chain  # noqa: E402

app = FastAPI()


# Cache chains per model so we don't rebuild on every request
@lru_cache(maxsize=4)
def _get_chain_cached(model: str):
    logger.info(f"[boot] Building RAG chain for model={model}")
    return get_rag_chain(model)


@app.on_event("startup")
def _startup_log():
    logger.info("[startup] RETRIEVER_MODE=%s", os.getenv("RETRIEVER_MODE"))
    logger.info("[startup] HYBRID_DENSE_WEIGHT=%s", os.getenv("HYBRID_DENSE_WEIGHT"))
    logger.info("[startup] CHROMA_PERSIST_DIR=%s", os.getenv("CHROMA_PERSIST_DIR"))
    logger.info("[startup] CHROMA_COLLECTION=%s", os.getenv("CHROMA_COLLECTION"))
    try:
        cnt = vectorstore._collection.count()
        logger.info("[startup] Chroma chunk count=%s", cnt)
    except Exception as e:
        logger.warning("[startup] Could not count Chroma chunks: %s", e)


# --- Debug helper: verify backend process config quickly ---
@app.get("/whoami")
def whoami():
    info = {
        "retriever_mode": os.getenv("RETRIEVER_MODE"),
        "hybrid_dense_weight": os.getenv("HYBRID_DENSE_WEIGHT"),
        "retriever_k": os.getenv("RETRIEVER_K"),
        "retriever_fetch_k": os.getenv("RETRIEVER_FETCH_K"),
        "retriever_lambda": os.getenv("RETRIEVER_LAMBDA"),
        "bm25_available": _bm25_available,
        "chroma_persist_dir": os.getenv("CHROMA_PERSIST_DIR"),
        "chroma_collection": os.getenv("CHROMA_COLLECTION"),
    }
    try:
        info["chroma_chunk_count"] = vectorstore._collection.count()
    except Exception as e:
        info["chroma_chunk_count_error"] = str(e)
    return info


# --- Chat endpoint ---
@app.post("/chat", response_model=QueryResponse)
def chat(query_input: QueryInput):
    session_id = query_input.session_id or str(uuid.uuid4())
    logger.info(
        "Session ID: %s, User Query: %s, Model: %s",
        session_id, query_input.question, query_input.model.value
    )

    chat_history = get_chat_history(session_id)
    rag_chain = _get_chain_cached(query_input.model.value)

    result = rag_chain.invoke(
        {"input": query_input.question, "chat_history": chat_history},
        config={
            "tags": ["api", "rag", os.getenv("RETRIEVER_MODE", "dense")],
            "metadata": {"session_id": session_id, "model": query_input.model.value},
        },
    )
    answer = result.get("answer", "")

    insert_application_logs(session_id, query_input.question, answer, query_input.model.value)
    logger.info("Session ID: %s, AI Response: %s", session_id, answer)

    return QueryResponse(answer=answer, session_id=session_id, model=query_input.model)


# --- Document upload ---
@app.post("/upload-doc")
def upload_and_index_document(file: UploadFile = File(...)):
    allowed_extensions = [".pdf"]
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed types are: {', '.join(allowed_extensions)}")

    documents_dir = "data/documents"
    os.makedirs(documents_dir, exist_ok=True)
    temp_file_path = os.path.join(documents_dir, f"temp_{file.filename}")

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_id = insert_document_record(file.filename)
        success = index_document_to_chroma(temp_file_path, file_id, original_filename=file.filename)

        if success:
            return {"message": f"File {file.filename} has been successfully uploaded and indexed.", "file_id": file_id}
        else:
            delete_document_record(file_id)
            raise HTTPException(status_code=500, detail=f"Failed to index {file.filename}.")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# --- List documents ---
@app.get("/list-docs", response_model=list[DocumentInfo])
def list_documents():
    return get_all_documents()


# --- Delete document ---
@app.post("/delete-doc")
def delete_document(request: DeleteFileRequest):
    chroma_delete_success = delete_doc_from_chroma(request.file_id)

    if chroma_delete_success:
        db_delete_success = delete_document_record(request.file_id)
        if db_delete_success:
            return {"message": f"Successfully deleted document with file_id {request.file_id} from the system."}
        else:
            return {"error": f"Deleted from Chroma but failed to delete document with file_id {request.file_id} from the database."}
    else:
        return {"error": f"Failed to delete document with file_id {request.file_id} from Chroma."}
