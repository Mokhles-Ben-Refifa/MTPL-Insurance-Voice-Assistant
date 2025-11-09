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

# ---------- Modern Professional Styling ----------
SIDEBAR_CSS = """
<style>
/* Sidebar base styling with subtle gradient */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #fafbfc 0%, #ffffff 100%);
}
[data-testid="stSidebar"] > div {
    padding: 1.25rem 0.9rem;
}

/* Section titles - minimalist and elegant */
.sidebar-section-title {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #8b92a8;
    margin: 1.5rem 0 0.7rem 0.15rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #e8eaf0;
    position: relative;
}
.sidebar-section-title::before {
    content: '';
    position: absolute;
    bottom: -2px;
    left: 0;
    width: 40px;
    height: 2px;
    background: linear-gradient(90deg, #667eea, #764ba2);
}
.sidebar-section-title:first-child { margin-top: 0; }

/* New Chat Button - Primary CTA with icon */
.new-chat-btn .stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 1rem;
    font-weight: 600;
    font-size: 0.95rem;
    box-shadow: 0 4px 14px rgba(102, 126, 234, 0.3);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.new-chat-btn .stButton > button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    transition: left 0.5s;
}
.new-chat-btn .stButton > button:hover::before {
    left: 100%;
}
.new-chat-btn .stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
}

/* Chat list container - smooth scrolling */
.chat-list-container {
    display: flex;
    flex-direction: column;
    gap: 4px;
    max-height: 420px;
    overflow-y: auto;
    padding-right: 6px;
    margin-top: 0.5rem;
}
.chat-list-container > div {
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}
.chat-list-container .stContainer {
    padding: 0 !important;
    margin: 0 !important;
}

/* Custom scrollbar - refined */
.chat-list-container::-webkit-scrollbar { width: 5px; }
.chat-list-container::-webkit-scrollbar-track {
    background: transparent;
}
.chat-list-container::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #d1d5db, #9ca3af);
    border-radius: 10px;
}
.chat-list-container::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, #9ca3af, #6b7280);
}

/* Chat row - clean flex layout */
.chat-row {
    display: flex;
    align-items: center;
    gap: 6px;
    margin: 0 !important;
    padding: 0 !important;
    transition: all 0.2s ease;
}
.chat-row > div {
    margin: 0 !important;
    padding: 0 !important;
}
.chat-row .stColumns {
    gap: 6px !important;
}
.chat-row [data-testid="column"] {
    padding: 0 !important;
}

/* Chat card - modern card design */
.chat-row .stButton > button {
    width: 100%;
    text-align: left;
    border-radius: 11px;
    border: 1.5px solid #e8eaf0;
    background: #ffffff;
    padding: 9px 11px;
    min-height: 38px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    white-space: pre-line;
    position: relative;
    overflow: hidden;
    font-size: 0.875rem;
    line-height: 1.35;
    margin: 0 !important;
}

/* Active chat - vibrant accent */
.chat-row.active .stButton > button {
    background: linear-gradient(135deg, #f5f7ff 0%, #eef1ff 100%);
    border: 1.5px solid #667eea;
    box-shadow: 0 3px 10px rgba(102, 126, 234, 0.2);
    transform: translateX(3px);
}
.chat-row.active .stButton > button::before {
    content: '';
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 4px;
    background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    box-shadow: 2px 0 8px rgba(102, 126, 234, 0.3);
}

/* Hover effect - subtle lift */
.chat-row .stButton > button:hover {
    border-color: #c7cde0;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.1);
    transform: translateX(2px);
    background: #fafbff;
}

/* Chat text - typography refinement */
.chat-title {
    display: block;
    font-weight: 600;
    font-size: 0.9rem;
    color: #2d3748;
    margin-bottom: 0.2rem;
    line-height: 1.35;
}
.chat-preview {
    display: block;
    font-size: 0.78rem;
    color: #718096;
    line-height: 1.4;
    opacity: 0.9;
}

/* Delete button - elegant hover reveal */
.chat-row .del-wrap {
    opacity: 0;
    transition: all 0.25s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
}
.chat-row:hover .del-wrap { opacity: 1; }
.chat-row .del-wrap .stButton > button {
    padding: 7px;
    min-height: 38px;
    width: 38px;
    height: 38px;
    border-radius: 10px;
    border: 1.5px solid #fee2e2;
    background: #ffffff;
    color: #ef4444;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 1px 3px rgba(239, 68, 68, 0.1);
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 !important;
}
.chat-row .del-wrap .stButton > button:hover {
    background: #fef2f2;
    border-color: #fca5a5;
    transform: scale(1.08) rotate(5deg);
    box-shadow: 0 2px 8px rgba(239, 68, 68, 0.2);
}

/* Upload section - modern drop zone */
.upload-section {
    background: linear-gradient(135deg, #ffffff 0%, #f9fafb 100%);
    border: 2px dashed #c7cde0;
    border-radius: 14px;
    padding: 1.2rem;
    margin: 0.7rem 0;
    transition: all 0.3s ease;
    position: relative;
}
.upload-section::before {
    content: '📎';
    position: absolute;
    top: 0.8rem;
    right: 0.8rem;
    font-size: 1.5rem;
    opacity: 0.15;
}
.upload-section:hover {
    border-color: #667eea;
    background: linear-gradient(135deg, #fafbff 0%, #f5f7ff 100%);
    box-shadow: 0 2px 12px rgba(102, 126, 234, 0.1);
}
.upload-section .stFileUploader { padding: 0; }
.upload-section .stFileUploader > div {
    border: none !important;
    background: transparent !important;
}

/* Upload button - secondary action */
.upload-btn .stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1rem;
    font-weight: 600;
    font-size: 0.9rem;
    transition: all 0.25s ease;
    box-shadow: 0 2px 8px rgba(102, 126, 234, 0.25);
}
.upload-btn .stButton > button:hover {
    background: linear-gradient(135deg, #5568d3 0%, #6a3d93 100%);
    transform: translateY(-2px);
    box-shadow: 0 4px 14px rgba(102, 126, 234, 0.35);
}

/* Documents section - organized list */
.doc-list-container {
    max-height: 300px;
    overflow-y: auto;
    padding-right: 4px;
}
.doc-row {
    display: flex;
    gap: 8px;
    align-items: center;
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.5rem;
    background: #ffffff;
    border: 1.5px solid #e8eaf0;
    border-radius: 10px;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}
.doc-row:hover {
    background: linear-gradient(135deg, #fafbff 0%, #f9fafb 100%);
    border-color: #c7cde0;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.06);
    transform: translateX(2px);
}
.doc-row .doc-name {
    flex: 1;
    font-size: 0.86rem;
    color: #2d3748;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-weight: 500;
}
.doc-row .stButton > button {
    padding: 0.45rem 0.6rem;
    border-radius: 8px;
    border: 1.5px solid #fee2e2;
    background: #ffffff;
    color: #ef4444;
    min-width: 36px;
    font-size: 0.85rem;
    transition: all 0.2s ease;
}
.doc-row .stButton > button:hover {
    background: #fef2f2;
    border-color: #fca5a5;
    transform: scale(1.08) rotate(5deg);
    box-shadow: 0 2px 8px rgba(239, 68, 68, 0.2);
}

/* Refresh button - subtle utility action */
.refresh-btn .stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #f7f8fa 0%, #ffffff 100%);
    color: #6b7280;
    border: 1.5px solid #e8eaf0;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    font-weight: 500;
    font-size: 0.87rem;
    transition: all 0.25s ease;
    margin-bottom: 0.7rem;
}
.refresh-btn .stButton > button:hover {
    background: linear-gradient(135deg, #f3f4f6 0%, #f9fafb 100%);
    color: #374151;
    border-color: #c7cde0;
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
}

/* Divider - minimal and clean */
hr {
    margin: 1.5rem 0;
    border: none;
    border-top: 2px solid #e8eaf0;
    opacity: 1;
}

/* Empty state - friendly messaging */
[data-testid="stSidebar"] .stCaption {
    text-align: center;
    color: #a0aec0;
    font-style: italic;
    padding: 1rem 0.5rem;
    font-size: 0.85rem;
    background: linear-gradient(135deg, #fafbfc 0%, #f9fafb 100%);
    border-radius: 10px;
    border: 1.5px dashed #e8eaf0;
    margin: 0.5rem 0;
}

/* Success/info messages - polished notifications */
[data-testid="stSidebar"] .stSuccess {
    border-radius: 10px;
    font-size: 0.85rem;
    background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
    border: 1.5px solid #86efac;
    padding: 0.6rem 0.9rem;
    box-shadow: 0 2px 8px rgba(134, 239, 172, 0.15);
}

/* Animation for new elements */
@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateX(-10px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

.chat-row, .doc-row {
    animation: slideIn 0.3s ease-out;
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

    # New chat button with enhanced styling
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

    # Scrollable chat list container
    st.sidebar.markdown('<div class="chat-list-container">', unsafe_allow_html=True)

    for s in sessions:
        title = (s.get("title") or "").strip()

        # Remove common prefixes
        for prefix in ["New chat", "Untitled Chat", "new chat", "untitled chat", "New Conversation", "new conversation"]:
            if title.startswith(prefix):
                title = title[len(prefix):].strip()
                break

        title = title or "💬 Chat"

        preview = (s.get("last_message") or "").strip()
        if len(preview) > 65:
            preview = preview[:62] + "..."

        # Build label
        label = title
        if preview:
            label += f"\n{preview}"

        # active row class
        row_class = "chat-row active" if s["id"] == current else "chat-row"

        with st.sidebar.container():
            st.markdown(f'<div class="{row_class}">', unsafe_allow_html=True)

            # Center-align delete button with chat card
            try:
                c_open, c_del = st.columns([0.85, 0.15], vertical_alignment="center")
            except TypeError:
                c_open, c_del = st.columns([0.85, 0.15])

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
    st.sidebar.markdown('<div class="sidebar-section-title">📤 Upload Documents</div>', unsafe_allow_html=True)

    # Upload section with enhanced styling
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
                    upload_response = upload_document(uploaded_file)
                    if upload_response:
                        st.sidebar.success(f"✓ Document uploaded successfully!")
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
        # Lazy fetch on first render if absent
        try:
            st.session_state.documents = list_documents()
            docs = st.session_state.get("documents") or []
        except Exception:
            pass

    if not docs:
        st.sidebar.caption("📂 No documents uploaded yet.")
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