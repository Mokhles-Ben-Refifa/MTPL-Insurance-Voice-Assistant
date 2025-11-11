import streamlit as st
from typing import Optional
from src.api_utils import upload_document, list_documents, delete_document
from src.db_utils import (
    get_sessions, create_session,
    delete_session_and_logs, get_chat_history, initialize_database
)
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

# ---------- Styling (sessions + documents share the SAME row/card CSS) ----------


SIDEBAR_CSS = """
<style>
/* ---- Compact spacing variables ---- */
[data-testid="stSidebar"] {
  --chat-gap: 6px;        /* spacing between rows */
  --section-gap: 8px;     /* spacing between sections - REDUCED */
  --card-radius: 12px;
}

/* Sidebar base */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #fafbfc 0%, #ffffff 100%);
}
[data-testid="stSidebar"] > div {
    padding: 1rem 0.9rem;  /* Reduced top/bottom padding */
}

/* Section title - COMPACT */
.sidebar-section-title {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #8b92a8;
    margin: var(--section-gap) 0 0.5rem 0.15rem;  /* Reduced margins */
    padding-bottom: 0.3rem;  /* Reduced padding */
    border-bottom: 2px solid #e8eaf0;
    position: relative;
}
.sidebar-section-title::before {
    content: '';
    position: absolute;
    bottom: -2px; left: 0;
    width: 40px; height: 2px;
    background: linear-gradient(90deg, #667eea, #764ba2);
}
.sidebar-section-title:first-child { margin-top: 0; }

/* New Chat Button - COMPACT */
.new-chat-btn .stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white; border: none; border-radius: var(--card-radius);
    padding: 0.65rem 1rem;  /* Reduced padding */
    font-weight: 600; font-size: 0.95rem;
    box-shadow: 0 4px 14px rgba(102, 126, 234, 0.3);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    margin-bottom: 0.5rem;  /* Reduced margin */
    position: relative; overflow: hidden;
}
.new-chat-btn .stButton > button::before {
    content: ''; position: absolute; top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    transition: left 0.5s;
}
.new-chat-btn .stButton > button:hover::before { left: 100%; }
.new-chat-btn .stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
}

/* ===== Shared row system (used by both chats and docs) ===== */

/* Normalize Streamlit spacing inside the lists */
.row-list [data-testid="element-container"],
.row-list [data-testid="stVerticalBlock"],
.row-list [data-testid="stHorizontalBlock"],
.row-list [data-testid="stVerticalBlock"] > div,
.row-list .stButton,
.row-list .stButton > * {
  margin: 0 !important;
  padding-bottom: 0 !important;
}
.row-list [data-testid="stVerticalBlock"] { gap: 0 !important; }
.row-list [data-testid="stHorizontalBlock"] { gap: 6px !important; }
.row-list [data-testid="column"] { padding: 0 !important; }

/* List container (scroll + spacing) - COMPACT */
.row-list {
    display: flex;
    flex-direction: column;
    row-gap: var(--chat-gap);
    max-height: 300px;  /* Reduced height to show more sections */
    overflow-y: auto;
    padding-right: 6px;
    margin-top: 0.3rem;  /* Reduced margin */
}
.row-list::-webkit-scrollbar { width: 5px; }
.row-list::-webkit-scrollbar-track { background: transparent; }
.row-list::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #d1d5db, #9ca3af); border-radius: 10px;
}
.row-list::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, #9ca3af, #6b7280);
}

/* Row container */
.row {
    display: flex; align-items: center; gap: 6px;
    margin: 0 !important; padding: 0 !important;
    transition: all 0.2s ease;
}
.row > div { margin: 0 !important; padding: 0 !important; }
.row .stColumns { gap: 6px !important; }
.row [data-testid="column"] { padding: 0 !important; }

/* Card button (title) */
.row .stButton > button {
    width: 100%; text-align: left;
    border-radius: 11px; border: 1.5px solid #e8eaf0; background: #ffffff;
    padding: 9px 11px; min-height: 38px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    white-space: pre-line; position: relative; overflow: hidden;
    font-size: 0.875rem; line-height: 1.35;
    margin: 0 !important;
}

/* Active row highlight */
.row.active .stButton > button {
    background: linear-gradient(135deg, #f5f7ff 0%, #eef1ff 100%);
    border: 1.5px solid #667eea;
    box-shadow: 0 3px 10px rgba(102, 126, 234, 0.2);
    transform: translateX(3px);
}
.row.active .stButton > button::before {
    content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
    background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    box-shadow: 2px 0 8px rgba(102, 126, 234, 0.3);
}

/* Hover */
.row .stButton > button:hover {
    border-color: #c7cde0; box-shadow: 0 4px 14px rgba(0, 0, 0, 0.1);
    transform: translateX(2px); background: #fafbff;
}

/* Delete control (reveals on hover) */
.row .del-wrap {
    opacity: 0; transition: all 0.25s ease;
    display: flex; align-items: center; justify-content: center; height: 100%;
}
.row:hover .del-wrap { opacity: 1; }
.row .del-wrap .stButton > button {
    padding: 7px; min-height: 38px; width: 38px; height: 38px;
    border-radius: 10px; border: 1.5px solid #fee2e2; background: #ffffff; color: #ef4444;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 1px 3px rgba(239, 68, 68, 0.1);
    display: flex; align-items: center; justify-content: center; margin: 0 !important;
}
.row .del-wrap .stButton > button:hover {
    background: #fef2f2; border-color: #fca5a5;
    transform: scale(1.08) rotate(5deg);
    box-shadow: 0 2px 8px rgba(239, 68, 68, 0.2);
}

/* Upload & utility buttons - COMPACT */
.upload-section {
    background: linear-gradient(135deg, #ffffff 0%, #f9fafb 100%);
    border: 2px dashed #c7cde0; border-radius: 14px;
    padding: 0.8rem;  /* Reduced padding */
    margin: 0.4rem 0;  /* Reduced margin */
    transition: all 0.3s ease; position: relative;
}
.upload-section::before {
    content: '📎'; position: absolute; top: 0.6rem; right: 0.6rem;
    font-size: 1.5rem; opacity: 0.15;
}
.upload-section:hover {
    border-color: #667eea;
    background: linear-gradient(135deg, #fafbff 0%, #f5f7ff 100%);
    box-shadow: 0 2px 12px rgba(102, 126, 234, 0.1);
}
.upload-section .stFileUploader { padding: 0; }
.upload-section .stFileUploader > div { border: none !important; background: transparent !important; }

.refresh-btn .stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #f7f8fa 0%, #ffffff 100%);
    color: #6b7280; border: 1.5px solid #e8eaf0;
    border-radius: 10px; padding: 0.5rem 1rem;  /* Reduced padding */
    font-weight: 500; font-size: 0.87rem;
    transition: all 0.25s ease; 
    margin-bottom: 0.4rem;  /* Reduced margin */
}
.refresh-btn .stButton > button:hover {
    background: linear-gradient(135deg, #f3f4f6 0%, #f9fafb 100%);
    color: #374151; border-color: #c7cde0; transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
}

/* Divider & messages - COMPACT */
hr { 
    margin: 0.8rem 0;  /* Reduced from 1.5rem */
    border: none; 
    border-top: 2px solid #e8eaf0; 
    opacity: 1; 
}

[data-testid="stSidebar"] .stCaption {
    text-align: center; color: #a0aec0; font-style: italic;
    padding: 0.8rem 0.5rem;  /* Reduced padding */
    font-size: 0.85rem;
    background: linear-gradient(135deg, #fafbfc 0%, #f9fafb 100%);
    border-radius: 10px; border: 1.5px dashed #e8eaf0; 
    margin: 0.3rem 0;  /* Reduced margin */
}

[data-testid="stSidebar"] .stSuccess {
    border-radius: 10px; font-size: 0.85rem;
    background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
    border: 1.5px solid #86efac; 
    padding: 0.5rem 0.8rem;  /* Reduced padding */
    box-shadow: 0 2px 8px rgba(134, 239, 172, 0.15);
    margin: 0.3rem 0;  /* Added margin control */
}

/* Subtle entrance */
@keyframes slideIn { from { opacity: 0; transform: translateX(-10px);} to {opacity: 1; transform: translateX(0);} }
.row { animation: slideIn 0.3s ease-out; }
</style>
"""

def display_sidebar() -> None:
    if not st.session_state.get("db_initialized"):
        try:
            initialize_database()
        except Exception:
            pass
        st.session_state["db_initialized"] = True

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
    with st.sidebar.container():
        st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
        if st.button("✨ New Chat", use_container_width=True):
            _new_chat()
        st.markdown("</div>", unsafe_allow_html=True)

    sessions = get_sessions()
    if not sessions:
        st.sidebar.caption("💬 No conversations yet. Start your first chat!")
        return

    current = st.session_state.get("session_id")

    st.sidebar.markdown('<div class="row-list">', unsafe_allow_html=True)

    for s in sessions:
        title = (s.get("title") or "").strip()
        for prefix in ["New chat", "Untitled Chat", "new chat", "untitled chat", "New Conversation", "new conversation"]:
            if title.startswith(prefix):
                title = title[len(prefix):].strip()
                break
        title = title or "💬"

        preview = (s.get("last_message") or "").strip()
        if len(preview) > 65:
            preview = preview[:62] + "..."
        label = title + (f"\n{preview}" if preview else "")

        row_class = "row active" if s["id"] == current else "row"
        st.markdown(f'<div class="{row_class}">', unsafe_allow_html=True)

        try:
            c_open, c_del = st.sidebar.columns([0.85, 0.15], vertical_alignment="center")
        except TypeError:
            c_open, c_del = st.sidebar.columns([0.85, 0.15])

        with c_open:
            if st.button(label, key=f"open_{s['id']}", use_container_width=True):
                _open_session(s["id"])

        with c_del:
            st.markdown('<div class="del-wrap">', unsafe_allow_html=True)
            if st.button("🗑", key=f"del_{s['id']}", help="Delete conversation"):
                if delete_session_and_logs(s["id"]):
                    if st.session_state.get("session_id") == s["id"]:
                        left = get_sessions(limit=1)
                        _open_session(left[0]["id"] if left else create_session())
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.sidebar.markdown("</div>", unsafe_allow_html=True)

def _render_model_and_docs() -> None:
    st.sidebar.markdown('<div class="sidebar-section-title">📤 Upload Documents</div>', unsafe_allow_html=True)
    with st.sidebar.container():
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drop files or browse",
            type=["pdf", "docx", "html"],
            label_visibility="visible"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file:
        with st.sidebar.container():
            st.markdown('<div class="upload-btn">', unsafe_allow_html=True)
            if st.button("📤 Upload Document", use_container_width=True):
                with st.spinner("Uploading document..."):
                    if upload_document(uploaded_file):
                        st.sidebar.success("✓ Document uploaded successfully!")
                        st.session_state.documents = list_documents()
                        st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    st.sidebar.divider()
    st.sidebar.markdown('<div class="sidebar-section-title">📚 Document Library</div>', unsafe_allow_html=True)

    # Refresh button
    with st.sidebar.container():
        st.markdown('<div class="refresh-btn">', unsafe_allow_html=True)
        if st.button("🔄 Refresh Library", use_container_width=True):
            st.session_state.documents = list_documents()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    docs = st.session_state.get("documents") or []
    if not docs:
        try:
            st.session_state.documents = list_documents()
            docs = st.session_state.get("documents") or []
        except Exception:
            pass
    if not docs:
        st.sidebar.caption("📂 No documents uploaded yet.")
        return
    selected_doc_id = st.session_state.get("selected_doc_id")
    st.sidebar.markdown('<div class="row-list">', unsafe_allow_html=True)

    for doc in docs:
        fname = doc["filename"].strip()
        if len(fname) > 65:
            fname = fname[:62] + "..."
        label = f"📄 {fname}"

        row_class = "row active" if doc["id"] == selected_doc_id else "row"
        st.markdown(f'<div class="{row_class}">', unsafe_allow_html=True)

        try:
            col_title, col_del = st.sidebar.columns([0.85, 0.15], vertical_alignment="center")
        except TypeError:
            col_title, col_del = st.sidebar.columns([0.85, 0.15])

        with col_title:
            if st.button(label, key=f"doc_open_{doc['id']}", use_container_width=True):
                # Mark as selected; you can hook preview/inspect here if needed
                st.session_state["selected_doc_id"] = doc["id"]
                st.rerun()

        with col_del:
            st.markdown('<div class="del-wrap">', unsafe_allow_html=True)
            if st.button("🗑", key=f"doc_del_{doc['id']}", help="Delete document"):
                if delete_document(doc["id"]):
                    if st.session_state.get("selected_doc_id") == doc["id"]:
                        st.session_state["selected_doc_id"] = None
                    st.sidebar.success("✓ Document deleted!")
                    st.session_state.documents = list_documents()
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.sidebar.markdown("</div>", unsafe_allow_html=True)
