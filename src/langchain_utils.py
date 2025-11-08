from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import List
from langchain_core.documents import Document
import os
from src.chroma_utils import vectorstore
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

retriever = vectorstore.as_retriever(search_kwargs={"k": int(os.getenv("RETRIEVER_K", "5"))})

output_parser = StrOutputParser()

# Setting up prompts
contextualize_q_system_prompt = os.getenv("CONTEXTUALIZE_Q_PROMPT", 
    "Given a chat history and the latest user question " 
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# IMPROVED: More explicit prompt that forces the model to use context
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant. Answer the user's question using ONLY the information provided in the context below.

Context: {context}

IMPORTANT INSTRUCTIONS:
- If the context contains relevant information, use it to answer the question in detail.
- If the context does NOT contain relevant information, clearly state: "I don't have enough information in the provided documents to answer this question."
- DO NOT make up information or use knowledge outside the provided context.
- Always cite which part of the context you're using in your answer."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

def get_rag_chain(model="gemini-2.5-flash"):
    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=GOOGLE_API_KEY,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
    )
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)    
    return rag_chain

# DEBUG FUNCTION: Test retrieval directly
def test_retrieval(query: str, k: int = 5):
    """Test what documents are being retrieved for a query"""
    logger.info(f"Testing retrieval for query: {query}")
    
    # Check if vector store has any documents
    try:
        collection_count = vectorstore._collection.count()
        logger.info(f"Total documents in Chroma: {collection_count}")
        
        if collection_count == 0:
            logger.error("WARNING: No documents in vector store!")
            return []
    except Exception as e:
        logger.error(f"Error checking collection count: {e}")
    
    # Test retrieval
    docs = retriever.get_relevant_documents(query)
    logger.info(f"Retrieved {len(docs)} documents")
    
    for i, doc in enumerate(docs):
        logger.info(f"\n--- Document {i+1} ---")
        logger.info(f"Content preview: {doc.page_content[:200]}...")
        logger.info(f"Metadata: {doc.metadata}")
    
    return docs
