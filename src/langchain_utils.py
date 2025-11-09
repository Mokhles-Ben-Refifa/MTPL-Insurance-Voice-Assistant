from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from typing import List
import os, logging
from dotenv import load_dotenv
from src.chroma_utils import vectorstore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

# ---- Retrieval knobs (good defaults for multi-PDF) ----
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "6"))
RETRIEVER_FETCH_K = int(os.getenv("RETRIEVER_FETCH_K", "40"))
RETRIEVER_LAMBDA = float(os.getenv("RETRIEVER_LAMBDA", "0.5"))  # 0=more diverse
SEARCH_TYPE = os.getenv("RETRIEVER_SEARCH_TYPE", "mmr")         # "mmr" >> "similarity"

retriever = vectorstore.as_retriever(
    search_type=SEARCH_TYPE,
    search_kwargs={"k": RETRIEVER_K, "fetch_k": RETRIEVER_FETCH_K, "lambda_mult": RETRIEVER_LAMBDA},
)

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

def get_rag_chain(model: str = "gemini-2.5-flash"):
    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=GOOGLE_API_KEY,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
    )
    history_aware = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware, qa_chain)

# ---- Debug helper: see scores, filenames, pages ----
def test_retrieval(query: str, k: int = 10):
    logger.info(f"[debug] query={query}")
    try:
        total = vectorstore._collection.count()
        logger.info(f"[debug] total_vectors={total}")
    except Exception as e:
        logger.error(f"[debug] count error: {e}")

    rows = vectorstore.similarity_search_with_score(query, k=k)
    out = []
    for rank, (doc, score) in enumerate(rows, start=1):
        logger.info(f"[{rank}] score={score:.4f} file={doc.metadata.get('filename')} page={doc.metadata.get('page')} :: {doc.page_content[:120]!r}")
        out.append((doc, score))
    return out