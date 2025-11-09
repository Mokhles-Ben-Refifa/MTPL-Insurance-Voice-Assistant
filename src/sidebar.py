import streamlit as st
from typing import Optional
from src.api_utils import upload_document, list_documents, delete_document
from src.db_utils import (
    get_sessions, create_session,
    delete_session_and_logs, get_chat_history, initialize_database
)

# ---------- URL param helpers (support old/new Streamlit) ----------
def _get_sid_from_url() -> Optional[str]:
    qp = getattr(st, "query_params", None)
    if qp is not None:
        val = qp.get("sid")
        if isinstance(val, list):
            return val[0] if val else None
        return val
    params = st.experimental_get_query_params()
    return params.get("sid", [None])[0]

def _set_sid_in_url(session_id: str) -> None:
    qp = getattr(st, "query_params", None)
    if qp is not None:
        qp["sid"] = session_id
        return
    st.experimental_set_query_params(sid=session_id)

# ---------- Session resolution ----------
def _ensure_current_session() -> None:
    url_sid = _get_sid_from_url()
    if url_sid:
        st.session_state.session_id = url_sid
        st.session_state.messages = get_chat_history(url_sid)
        return

    if st.session_state.get("session_id"):
        return

    sessions = get_sessions(limit=1)
    if sessions:
        sid = sessions[0]["id"]
        st.session_state.session_id = sid
        st.session_state.messages = get_chat_history(sid)
        _set_sid_in_url(sid)
        return

    sid = create_session()
    st.session_state.session_id = sid
    st.session_state.messages = []
    _set_sid_in_url(sid)

def _open_session(session_id: str) -> None:
    st.session_state.session_id = session_id
    st.session_state.messages = get_chat_history(session_id)
    _set_sid_in_url(session_id)

def _new_chat() -> None:
    sid = create_session()
    _open_session(sid)

# ---------- Enhanced Professional Styling ----------
SIDEBAR_CSS = """
<style>
/* Sidebar background and spacing */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
}

[data-testid="stSidebar"] > div {
    padding: 1.5rem 1rem;
}

/* Section titles - more refined */
.sidebar-section-title {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6b7280;
    margin: 1.5rem 0 0.75rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #e5e7eb;
}

.sidebar-section-title:first-child {
    margin-top: 0;
}

/* New Chat Button - Primary action style */
.new-chat-btn .stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    font-weight: 600;
    font-size: 0.95rem;
    box-shadow: 0 4px 6px rgba(102, 126, 234, 0.25);
    transition: all 0.2s ease;
    margin-bottom: 1rem;
}

.new-chat-btn .stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(102, 126, 234, 0.35);
}

/* Chat list container */
.chat-list-container {
    max-height: 400px;
    overflow-y: auto;
    padding-right: 4px;
}

/* Custom scrollbar */
.chat-list-container::-webkit-scrollbar {
    width: 6px;
}

.chat-list-container::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 10px;
}

.chat-list-container::-webkit-scrollbar-thumb {
    background: #d1d5db;
    border-radius: 10px;
}

.chat-list-container::-webkit-scrollbar-thumb:hover {
    background: #9ca3af;
}

/* Chat row layout */
.chat-row {
    display: flex;
    gap: 6px;
    align-items: stretch;
    margin-bottom: 0.5rem;
}

/* Chat card design */
.chat-row .stButton > button {
    width: 100%;
    text-align: left;
    border-radius: 10px;
    border: 1px solid #e5e7eb;
    background: #ffffff;
    padding: 0.85rem 1rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    transition: all 0.2s ease;
    white-space: pre-line;
    position: relative;
    overflow: hidden;
}

/* Active chat - elegant accent */
.chat-row.active .stButton > button {
    background: linear-gradient(135deg, #f0f4ff 0%, #e8eeff 100%);
    border: 1px solid #667eea;
    box-shadow: 0 2px 8px rgba(102, 126, 234, 0.15);
}

.chat-row.active .stButton > button::before {
    content: '';
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 4px;
    background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
}

/* Hover effect */
.chat-row .stButton > button:hover {
    border-color: #d1d5db;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    transform: translateX(2px);
}

/* Chat text styling */
.chat-title {
    display: block;
    font-weight: 600;
    font-size: 0.9rem;
    color: #1f2937;
    margin-bottom: 0.25rem;
    line-height: 1.3;
}

.chat-preview {
    display: block;
    font-size: 0.8rem;
    color: #6b7280;
    line-height: 1.4;
}

/* Delete button - hidden by default */
.chat-row .del-wrap {
    opacity: 0;
    transition: opacity 0.2s ease;
}

.chat-row:hover .del-wrap {
    opacity: 1;
}

.chat-row .del-wrap .stButton > button {
    padding: 0.85rem 0.6rem;
    border-radius: 10px;
    border: 1px solid #fee2e2;
    background: #ffffff;
    color: #ef4444;
    min-width: 42px;
    transition: all 0.2s ease;
}

.chat-row .del-wrap .stButton > button:hover {
    background: #fef2f2;
    border-color: #fecaca;
    transform: scale(1.05);
}

/* Upload section */
.upload-section {
    background: #ffffff;
    border: 2px dashed #d1d5db;
    border-radius: 12px;
    padding: 1.25rem;
    margin: 0.75rem 0;
    transition: all 0.2s ease;
}

.upload-section:hover {
    border-color: #667eea;
    background: #fafbff;
}

.upload-section .stFileUploader {
    padding: 0;
}

.upload-section .stFileUploader > div {
    border: none !important;
    background: transparent !important;
}

/* Upload button */
.upload-btn .stButton > button {
    width: 100%;
    background: #667eea;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.65rem 1rem;
    font-weight: 600;
    font-size: 0.9rem;
    transition: all 0.2s ease;
}

.upload-btn .stButton > button:hover {
    background: #5568d3;
    transform: translateY(-1px);
}

/* Documents section */
.doc-list-container {
    max-height: 300px;
    overflow-y: auto;
}

.doc-row {
    display: flex;
    gap: 8px;
    align-items: center;
    padding: 0.7rem 0.85rem;
    margin-bottom: 0.5rem;
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    transition: all 0.2s ease;
}

.doc-row:hover {
    background: #f9fafb;
    border-color: #d1d5db;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
}

.doc-row .doc-name {
    flex: 1;
    font-size: 0.875rem;
    color: #374151;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-weight: 500;
}

.doc-row .stButton > button {
    padding: 0.4rem 0.55rem;
    border-radius: 6px;
    border: 1px solid #fee2e2;
    background: #ffffff;
    color: #ef4444;
    min-width: 36px;
    font-size: 0.85rem;
    transition: all 0.2s ease;
}

.doc-row .stButton > button:hover {
    background: #fef2f2;
    border-color: #fecaca;
    transform: scale(1.05);
}

/* Refresh button */
.refresh-btn .stButton > button {
    width: 100%;
    background: #f3f4f6;
    color: #6b7280;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-weight: 500;
    font-size: 0.875rem;
    transition: all 0.2s ease;
    margin-bottom: 0.75rem;
}

.refresh-btn .stButton > button:hover {
    background: #e5e7eb;
    color: #374151;
}

/* Divider */
hr {
    margin: 1.5rem 0;
    border: none;
    border-top: 1px solid #e5e7eb;
    opacity: 1;
}

/* Empty state messages */
[data-testid="stSidebar"] .stCaption {
    text-align: center;
    color: #9ca3af;
    font-style: italic;
    padding: 1rem 0;
}

/* Success/info messages */
[data-testid="stSidebar"] .stSuccess {
    border-radius: 8px;
    font-size: 0.85rem;
}
</style>
"""

def display_sidebar() -> None:
    # one-time DB init
    if not st.session_state.get("db_initialized"):
        try:
            initialize_database()
        except Exception:
            pass
        st.session_state["db_initialized"] = True

    # CSS inject once
    if not st.session_state.get("sidebar_css_injected"):
        st.sidebar.markdown(SIDEBAR_CSS, unsafe_allow_html=True)
        st.session_state["sidebar_css_injected"] = True

    _ensure_current_session()
    _render_chat_list()
    st.sidebar.divider()
    _render_model_and_docs()

# ---------- Sidebar UI blocks ----------
def _render_chat_list() -> None:
    st.sidebar.markdown('<div class="sidebar-section-title">Conversations</div>', unsafe_allow_html=True)

    # New chat button with primary styling
    with st.sidebar.container():
        st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
        if st.button("✨ New Chat", use_container_width=True):
            _new_chat()
        st.markdown("</div>", unsafe_allow_html=True)

    sessions = get_sessions()

    if not sessions:
        st.sidebar.caption("No conversations yet. Start a new chat!")
        return

    current = st.session_state.get("session_id")

    # Scrollable chat list container
    st.sidebar.markdown('<div class="chat-list-container">', unsafe_allow_html=True)
    
    for s in sessions:
        title = (s.get("title") or "").strip()
        
        # Remove common prefixes
        for prefix in ["New chat", "Untitled Chat", "new chat", "untitled chat", "New Conversation", "new conversation"]:
            if title.startswith(prefix):
                title = title[len(prefix):].strip()
                break
        
        # If empty, just skip showing title
        if not title:
            title = ""
        
        preview = (s.get("last_message") or "").strip()
        if len(preview) > 70:
            preview = preview[:67] + "..."
        
        # Build simple text label (buttons don't support HTML)
        label = f"💬 {title}"
        if preview:
            label += f"\n{preview}"

        # active row class
        row_class = "chat-row active" if s["id"] == current else "chat-row"

        with st.sidebar.container():
            st.markdown(f'<div class="{row_class}">', unsafe_allow_html=True)

            c_open, c_del = st.columns([1, 0.15])

            with c_open:
                if st.button(label, key=f"open_{s['id']}", use_container_width=True):
                    _open_session(s["id"])

            with c_del:
                st.markdown('<div class="del-wrap">', unsafe_allow_html=True)
                if st.button("🗑", key=f"del_{s['id']}", help="Delete conversation"):
                    deleted = delete_session_and_logs(s["id"])
                    if deleted:
                        if st.session_state.get("session_id") == s["id"]:
                            left = get_sessions(limit=1)
                            if left:
                                _open_session(left[0]["id"])
                            else:
                                _open_session(create_session())
                        st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
    
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

def _render_model_and_docs() -> None:
    st.sidebar.markdown('<div class="sidebar-section-title">Upload Documents</div>', unsafe_allow_html=True)

    # Upload section with enhanced styling
    with st.sidebar.container():
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drop file here or browse",
            type=["pdf", "docx", "html"],
            label_visibility="visible"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file:
        with st.sidebar.container():
            st.markdown('<div class="upload-btn">', unsafe_allow_html=True)
            if st.button("📤 Upload", use_container_width=True):
                with st.spinner("Uploading document..."):
                    upload_response = upload_document(uploaded_file)
                    if upload_response:
                        st.sidebar.success(f"✓ Document uploaded successfully!")
                        st.session_state.documents = list_documents()
                        st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    st.sidebar.divider()
    st.sidebar.markdown('<div class="sidebar-section-title">Document Library</div>', unsafe_allow_html=True)

    # Refresh button
    with st.sidebar.container():
        st.markdown('<div class="refresh-btn">', unsafe_allow_html=True)
        if st.button("🔄 Refresh", use_container_width=True):
            st.session_state.documents = list_documents()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    docs = st.session_state.get("documents") or []

    if not docs:
        # Lazy fetch on first render if absent
        try:
            st.session_state.documents = list_documents()
            docs = st.session_state.get("documents") or []
        except Exception:
            pass

    if not docs:
        st.sidebar.caption("No documents uploaded yet.")
        return

    # Scrollable document list
    st.sidebar.markdown('<div class="doc-list-container">', unsafe_allow_html=True)
    
    for doc in docs:
        with st.sidebar.container():
            st.markdown('<div class="doc-row">', unsafe_allow_html=True)
            st.markdown(f'<div class="doc-name">📄 {doc["filename"]}</div>', unsafe_allow_html=True)
            if st.button("🗑", key=f"doc_del_{doc['id']}", help="Delete document"):
                ok = delete_document(doc["id"])
                if ok:
                    st.sidebar.success("✓ Document deleted!")
                    st.session_state.documents = list_documents()
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
    
    st.sidebar.markdown("</div>", unsafe_allow_html=True)