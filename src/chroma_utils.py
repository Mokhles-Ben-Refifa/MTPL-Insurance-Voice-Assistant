from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List
import os, logging, hashlib

logging.basicConfig(
    filename=os.path.join(os.path.dirname(__file__), "..", "app.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---- Embeddings (multilingual) ----
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# ---- Persist dir + named collection ----
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
persist_dir = os.path.join(project_root, "data", "chroma_db")
os.makedirs(persist_dir, exist_ok=True)
collection_name = os.getenv("CHROMA_COLLECTION", "mtpl_docs_v1_minilm12")

logging.info(f"[chroma] persist_dir={persist_dir} collection={collection_name}")
try:
    tf = os.path.join(persist_dir, ".__wtest")
    with open(tf, "w") as f: f.write("ok")
    os.remove(tf)
except Exception as e:
    logging.error(f"[chroma] persist_dir not writable: {e}")
    raise

vectorstore = Chroma(
    collection_name=collection_name,
    persist_directory=persist_dir,
    embedding_function=embedding_function
)

# ---- Chunking ----
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

def load_and_split_document(file_path: str) -> List[Document]:
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith('.html'):
        loader = UnstructuredHTMLLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    docs = loader.load()
    splits = text_splitter.split_documents(docs)
    logging.info(f"[index] {file_path}: pages={len(docs)} -> chunks={len(splits)}")
    return splits

def _stable_id(filename: str, page: int | None, i: int, text: str) -> str:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:16]
    return f"{filename}::p{page if page is not None else -1}::i{i}::{h}"

def index_document_to_chroma(file_path: str, file_id: str, original_filename: str | None = None) -> bool:
    try:
        splits = load_and_split_document(file_path)
        meta_filename = original_filename or os.path.basename(file_path)

        ids = []
        for i, s in enumerate(splits):
            s.metadata["file_id"] = file_id
            s.metadata["source"] = file_path
            s.metadata["filename"] = meta_filename
            ids.append(_stable_id(meta_filename, s.metadata.get("page"), i, s.page_content))

        vectorstore.add_documents(splits, ids=ids)
        try: vectorstore.persist()
        except Exception: pass

        logging.info(f"[index] added={len(splits)} total={vectorstore._collection.count()}")
        return True
    except Exception as e:
        logging.error(f"[index] error {file_path}: {e}")
        return False

def delete_doc_from_chroma(file_id: str) -> bool:
    try:
        vectorstore._collection.delete(where={"file_id": file_id})
        try: vectorstore.persist()
        except Exception: pass
        logging.info(f"[delete] file_id={file_id} total={vectorstore._collection.count()}")
        return True
    except Exception as e:
        logging.error(f"[delete] error file_id={file_id}: {e}")
        return False
