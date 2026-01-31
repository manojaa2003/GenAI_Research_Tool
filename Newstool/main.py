from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# ---------------- CONSTANTS ----------------
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "news"

MAX_RECENT_MESSAGES = 4
SUMMARY_TRIGGER = 8

# ---------------- EMBEDDINGS ----------------
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"trust_remote_code": True}
)

# ---------------- LLM ----------------
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.9,
    max_tokens=500
)

# ---------------- VECTOR STORE ----------------
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings
)

# ---------------- INGESTION ----------------
def process_urls(urls):
    yield "Resetting vector store...✅"
    vector_store.reset_collection()

    yield "Loading data from URLs...✅"
    loader = UnstructuredURLLoader(urls=urls)
    documents = loader.load()

    yield "Splitting documents into chunks...✅"
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE
    )
    docs = splitter.split_documents(documents)

    yield "Storing chunks in vector database...✅"
    vector_store.add_documents(docs)

    yield "Ingestion complete 🎉"

# ---------------- MEMORY SUMMARIZATION ----------------
SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    """
    Summarize the following conversation.
    Keep only important facts and user intent.

    Conversation:
    {conversation}
    """
)

def summarize_conversation(messages):
    conversation_text = "\n".join(messages)
    chain = SUMMARY_PROMPT | llm | StrOutputParser()
    return chain.invoke({"conversation": conversation_text})

# ---------------- QA WITH MEMORY ----------------
def generate_answer(query, summary="", recent_msgs=None):
    if recent_msgs is None:
        recent_msgs = []

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_template(
        """
        You are a real estate news analyst.

        Conversation summary:
        {summary}

        Recent conversation:
        {chat_history}

        Article context:
        {context}

        Question:
        {query}
        """
    )

    chat_history = "\n".join(recent_msgs)

    chain = (
        {
            "context": retriever,
            "query": RunnablePassthrough(),
            "summary": lambda _: summary,
            "chat_history": lambda _: chat_history,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(query)

    # -------- MEMORY UPDATE (returned to frontend) --------
    recent_msgs.append(f"User: {query}")
    recent_msgs.append(f"Assistant: {answer}")

    if len(recent_msgs) > SUMMARY_TRIGGER:
        summary = summarize_conversation(recent_msgs)
        recent_msgs[:] = recent_msgs[-MAX_RECENT_MESSAGES * 2 :]

    return answer, summary, recent_msgs

# ---------------- LOCAL TEST ----------------
if __name__ == "__main__":
    urls = [
        "https://therealdeal.com/national/2026/01/28/alexander-brothers-sex-trafficking-trial-day-two/"
    ]

    for status in process_urls(urls):
        print(status)

    recent_msgs = []
    summary = ""

    ans, summary, recent_msgs = generate_answer(
        "What is this news about?",
        summary,
        recent_msgs
    )

    print(ans)
