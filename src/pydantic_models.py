# src/pydantic_models.py
from __future__ import annotations

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
from enum import Enum
from datetime import datetime

class ModelName(str, Enum):
    GEMINI_2_5_FLASH = "gemini-2.5-flash"

class QueryInput(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    # Optional so None is valid when starting a new chat
    session_id: Optional[str] = Field(default=None, description="Session ID if resuming an existing chat")
    model: ModelName = Field(default=ModelName.GEMINI_2_5_FLASH)

class Citation(BaseModel):
    filename: Optional[str] = Field(default=None, description="Source filename")
    page: Optional[int] = Field(default=None, description="Page number if available")
    score: Optional[float] = Field(default=None, description="Retriever score, if provided")
    snippet: Optional[str] = Field(default=None, description="Short excerpt for UX")

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    model: ModelName
    # Optional so your current responses still validate without including citations
    citations: Optional[List[Citation]] = None
    # Ignore any unexpected fields you might add later
    model_config = ConfigDict(extra="ignore")

class DocumentInfo(BaseModel):
    id: str
    filename: str
    upload_timestamp: datetime

class DeleteFileRequest(BaseModel):
    file_id: str
