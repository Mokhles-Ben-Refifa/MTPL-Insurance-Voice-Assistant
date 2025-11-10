# src/langchain_utils.py

from __future__ import annotations

import os
import re
import unicodedata
import logging
from typing import List, Optional, Any, Tuple

from dotenv import load_dotenv

# LangChain / LLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Retrievers & docs
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

# Pydantic v2 helpers (BaseRetriever is a Pydantic model)
from pydantic import Field, ConfigDict

# Cross-encoder (reranker)
from sentence_transformers import CrossEncoder

# Your dense vector store (Chroma)
from src.chroma_utils import vectorstore
from langfuse.langchain import CallbackHandler  
langfuse_handler = CallbackHandler()


# Logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Env / keys

load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")


# Retrieval knobs (env-tunable)

RETRIEVER_K = int(os.getenv("RETRIEVER_K", "6"))
RETRIEVER_FETCH_K = int(os.getenv("RETRIEVER_FETCH_K", "40"))
RETRIEVER_LAMBDA = float(os.getenv("RETRIEVER_LAMBDA", "0.5")) 
SEARCH_TYPE = os.getenv("RETRIEVER_SEARCH_TYPE", "mmr")         

RETRIEVER_MODE = os.getenv("RETRIEVER_MODE", "dense").lower()   
HYBRID_DENSE_WEIGHT = float(os.getenv("HYBRID_DENSE_WEIGHT", "0.55"))
BM25_MAX_DOCS = int(os.getenv("BM25_MAX_DOCS", "5000"))

# Reranker controls
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_TOP_N = int(os.getenv("RERANKER_TOP_N", str(RETRIEVER_K)))


_DENSE_FILE_FILTER = os.getenv("DENSE_FILENAME_FILTER") 


# Dense retriever (Chroma)

_dense_kwargs = {"k": RETRIEVER_K, "fetch_k": RETRIEVER_FETCH_K, "lambda_mult": RETRIEVER_LAMBDA}
if _DENSE_FILE_FILTER:
    allowed = [x.strip() for x in _DENSE_FILE_FILTER.split(",") if x.strip()]
    _dense_kwargs["filter"] = {"filename": {"$in": allowed}}
    logger.info("[dense] Applying filename filter to dense retrieval: %s", allowed)

dense_retriever = vectorstore.as_retriever(
    search_type=SEARCH_TYPE,
    search_kwargs=_dense_kwargs,
)


# BM25 builder (accent-insensitive)

_BM25: Optional[BM25Retriever] = None

def _fold_accents(text: str) -> str:
    """Remove accents: Zöldkártya -> zoldkartya."""
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))

def _bm25_tokenize(text: str) -> List[str]:
    
    t = _fold_accents(text).lower()
    return re.findall(r"[a-z0-9]+", t)

def _build_bm25_from_chroma(limit: Optional[int] = None) -> Optional[BM25Retriever]:
    """Create an in-memory BM25 index from the chunks stored in Chroma."""
    try:
        total = vectorstore._collection.count()
        take = min(total, limit or total)

        raw = vectorstore._collection.get(limit=take, include=["documents", "metadatas"])
        docs: List[Document] = []
        for content, meta in zip(raw.get("documents", []), raw.get("metadatas", [])):
            if not content:
                continue
            docs.append(Document(page_content=content, metadata=meta or {}))

        if not docs:
            logger.warning("[bm25] No documents pulled from Chroma; BM25 disabled.")
            return None

        bm25 = BM25Retriever.from_documents(docs, preprocess_func=_bm25_tokenize)
        bm25.k = RETRIEVER_K
        logger.info("[bm25] Built BM25 index over %d chunks (k=%d).", len(docs), bm25.k)
        return bm25
    except Exception as e:
        logger.warning("[bm25] Failed to build BM25 from Chroma: %s", e)
        return None

if RETRIEVER_MODE in {"bm25", "hybrid"}:
    _BM25 = _build_bm25_from_chroma(limit=BM25_MAX_DOCS)


# Bilingual query expansion

def expand_query(q: str) -> str:
    """
    Lightweight bilingual expansion for your domain so English questions
    will match Hungarian content in BM25 and help dense too.
    """
    ql = q.lower()
    extras: List[str] = []
    
    if "country" in ql or "valid" in ql:
        extras += ["Hol érvényes a biztosításom?", "Európai Gazdasági Térség", "Svájc", "Zöldkártya"]
    if "green card" in ql:
        extras += ["Zöldkártya"]
    if "eea" in ql or "european economic area" in ql:
        extras += ["Európai Gazdasági Térség"]

    #
    folded = [_fold_accents(t) for t in extras]
    
    return " ".join([q] + extras + folded)


# Expanding retriever wrapper (Pydantic-compliant)

class ExpandingRetriever(BaseRetriever):
    """Wrap any retriever to expand user query prior to retrieval (sync + async)."""

    
    inner: BaseRetriever = Field(...)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        qx = expand_query(query)
        
        if hasattr(self.inner, "invoke"):
            return self.inner.invoke(qx)
        
        return self.inner.get_relevant_documents(qx)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        qx = expand_query(query)
        if hasattr(self.inner, "ainvoke"):
            return await self.inner.ainvoke(qx)
        return await self.inner.aget_relevant_documents(qx)


# Optional Cross-Encoder Rerank (implemented as a retriever wrapper to avoid missing LC imports)

class CrossEncoderRerankRetriever(BaseRetriever):
    """
    Wrap a retriever, then rerank its results with a sentence-transformers CrossEncoder.
    Keeps the top_n most relevant documents.
    """
    inner: BaseRetriever = Field(...)
    ce: Any = Field(...)                     
    top_n: int = Field(default=RETRIEVER_K) 
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        docs = self.inner.invoke(query) if hasattr(self.inner, "invoke") else self.inner.get_relevant_documents(query)
        if not docs:
            return []
        pairs = [[query, d.page_content] for d in docs]
        try:
            scores = self.ce.predict(pairs)
        except Exception:
            # Some CE versions use .predict, others return numpy; handle both
            scores = self.ce.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
        return [doc for doc, _ in ranked[: self.top_n]]

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager)


# Choose active retriever (dense / bm25 / hybrid), then wrap with Expanding + optional Rerank

def _make_base_retriever() -> BaseRetriever:
    if RETRIEVER_MODE == "bm25" and _BM25:
        logger.info("[retriever] Using BM25 only.")
        return _BM25

    if RETRIEVER_MODE == "hybrid" and _BM25:
        weights = [HYBRID_DENSE_WEIGHT, 1.0 - HYBRID_DENSE_WEIGHT]
        logger.info("[retriever] Using HYBRID (dense/bm25) weights=%s", weights)
        return EnsembleRetriever(retrievers=[dense_retriever, _BM25], weights=weights)

    if RETRIEVER_MODE in {"bm25", "hybrid"} and not _BM25:
        logger.warning("[retriever] BM25 requested but unavailable; falling back to dense retriever.")

    logger.info("[retriever] Using DENSE only.")
    return dense_retriever

_base = _make_base_retriever()
active_retriever: BaseRetriever = ExpandingRetriever(inner=_base)  

# Optional: add reranker on top
if RERANKER_ENABLED:
    try:
        _ce = CrossEncoder(RERANKER_MODEL)  
        active_retriever = CrossEncoderRerankRetriever(inner=active_retriever, ce=_ce, top_n=RERANKER_TOP_N)
        logger.info("[rerank] enabled model=%s top_n=%s", RERANKER_MODEL, RERANKER_TOP_N)
    except Exception as e:
        logger.warning("[rerank] disabled: %s", e)


# Prompts

output_parser = StrOutputParser()

contextualize_q_system_prompt = os.getenv(
    "CONTEXTUALIZE_Q_PROMPT",
    "Given a chat history and the latest user question which might reference context in the chat history, "
    "formulate a standalone question which can be understood without the chat history. "
    "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant. Answer the user's question using ONLY the information provided in the context below.

Context:
{context}

IMPORTANT INSTRUCTIONS:
- If the context contains relevant information, use it to answer the question in detail.
- If the context does NOT contain relevant information, clearly state: "I don't have enough information in the provided documents to answer this question."
- DO NOT make up information or use knowledge outside the provided context.
- Always cite which part of the context you're using in your answer (filename and page if available)."""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])


# Public: build the full RAG chain

def get_rag_chain(model: str = "gemini-2.5-flash"):
    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=GOOGLE_API_KEY,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
    )
    history_aware = create_history_aware_retriever(llm, active_retriever, contextualize_q_prompt)
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    chain = create_retrieval_chain(history_aware, qa_chain)

    return chain.with_config({
        "run_name": "RAG QA Chain",
        "callbacks": [langfuse_handler],
    })


# Debug helpers

def test_dense_retrieval(query: str, k: int = 10) -> List[Tuple[Document, float]]:
    """Show dense (vector) hits with scores, filenames, and pages."""
    logger.info("[debug/dense] query=%s", query)
    try:
        total = vectorstore._collection.count()
        logger.info("[debug/dense] total_vectors=%s", total)
    except Exception as e:
        logger.error("[debug/dense] count error: %s", e)

    rows = vectorstore.similarity_search_with_score(query, k=k)
    out: List[Tuple[Document, float]] = []
    for rank, (doc, score) in enumerate(rows, start=1):
        logger.info("[%d] score=%.4f file=%s page=%s :: %r",
                    rank, score, doc.metadata.get("filename"), doc.metadata.get("page"),
                    doc.page_content[:120])
        out.append((doc, score))
    return out

def test_bm25_retrieval(query: str, k: int = 10):
    """Show BM25 hits using the new .invoke() API."""
    if not _BM25:
        logger.warning("[debug/bm25] BM25 not available.")
        return []
    _BM25.k = k
    qx = expand_query(query)  # make tests mimic production
    docs = _BM25.invoke(qx)
    for rank, doc in enumerate(docs[:k], start=1):
        logger.info("[%d] file=%s page=%s :: %r",
                    rank, doc.metadata.get("filename"), doc.metadata.get("page"),
                    doc.page_content[:120])
    return docs[:k]
