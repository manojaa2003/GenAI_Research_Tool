import streamlit as st
from main import process_urls, generate_answer

st.set_page_config(
    page_title="Real Estate Research Tool",
    page_icon="🏢",
    layout="wide"
)

st.markdown("""
<style>
.chat-box {
    border-radius: 12px;
    padding: 10px;
}
.sidebar-note {
    font-size: 13px;
    color: #d9534f;
}
.hero {
    background: linear-gradient(90deg, #1f2937, #111827);
    padding: 25px;
    border-radius: 16px;
    color: white;
}
.hero h1 {
    font-size: 40px;
}
.hero p {
    font-size: 16px;
    color: #d1d5db;
}
.badge {
    background-color: #2563eb;
    color: white;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 13px;
    display: inline-block;
    margin-right: 6px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <span class="badge">RAG • LangChain • Groq • Streamlit</span>
    <span class="badge">Source: The Real Deal</span>
    <span class="badge">1–3 Articles</span>
    <h1>🏢 Real Estate Research Tool</h1>
    <p>
        AI-powered question answering over real estate news articles using
        Retrieval-Augmented Generation (RAG).
    </p>
</div>
""", unsafe_allow_html=True)

st.write("")

st.sidebar.markdown("""

📌 <b>Supported Source</b><br>
<i>The Real Deal</i> only<br>
<span style="color:#6b7280;">https://therealdeal.com/...</span><br><br>
</div>
""", unsafe_allow_html=True)

st.sidebar.header("🔗 Article Ingestion")

# -------- URL INPUTS WITH PLACEHOLDERS --------
url1 = st.sidebar.text_input(
    "Article URL 1",
    placeholder="https://therealdeal.com/..."
)

url2 = st.sidebar.text_input(
    "Article URL 2 (optional)",
    placeholder="Leave empty if not needed"
)

url3 = st.sidebar.text_input(
    "Article URL 3 (optional)",
    placeholder="Leave empty if not needed"
)

status_box = st.sidebar.empty()

if st.sidebar.button("🚀 Process Articles"):
    urls = [u for u in (url1, url2, url3) if u.strip()]
    if not urls:
        status_box.error("Please provide at least one valid The Real Deal URL")
    else:
        for status in process_urls(urls):
            status_box.success(status)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    👨‍💻 **Built With**
    - LangChain
    - ChromaDB
    - Groq LLM
    - Streamlit
    """
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "summary" not in st.session_state:
    st.session_state.summary = ""

if "recent_messages" not in st.session_state:
    st.session_state.recent_messages = []

# ---------------- CHAT SECTION ----------------
st.markdown("## 💬 Ask Questions About the Article")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input(
    "Ask about people, cases, companies, or events mentioned in the article..."
)

if query:
    try:
        with st.chat_message("user"):
            st.markdown(query)

        st.session_state.messages.append(
            {"role": "user", "content": query}
        )

        answer, new_summary, new_recent_msgs = generate_answer(
            query=query,
            summary=st.session_state.summary,
            recent_msgs=st.session_state.recent_messages
        )

        st.session_state.summary = new_summary
        st.session_state.recent_messages = new_recent_msgs

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

    except Exception:
        st.warning("⚠️ Please process a *The Real Deal* article first")

st.markdown(
    "<hr><p style='font-size:11px;color:#6b7280;text-align:center;'>"
    "Currently supports The Real Deal articles • Multi-document analysis (1–3 URLs per session)"
    "</p>",
    unsafe_allow_html=True
)
