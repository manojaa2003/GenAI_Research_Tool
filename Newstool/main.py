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

CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "news"

MAX_RECENT_MESSAGES = 6
SUMMARY_TRIGGER = 12

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"trust_remote_code": True}
)

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.9,
    max_tokens=500
)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings
)

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

SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    """
    You are summarizing a conversation for long-term memory.

    Task:
    - Extract only important facts, decisions, entities, and user intent.
    - Remove greetings, small talk, and repeated information.
    - Do NOT infer, assume, or add new information.
    - Use ONLY what is explicitly stated in the conversation.

    If no meaningful information is present, return an empty summary.

    Conversation:
    {conversation}

    Summary:
    """
)

def summarize_conversation(messages):
    conversation_text = "\n".join(messages)
    chain = SUMMARY_PROMPT | llm | StrOutputParser()
    return chain.invoke({"conversation": conversation_text})

def generate_answer(query, summary="", recent_msgs=None):
    if recent_msgs is None:
        recent_msgs = []

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_template(
        """
        You are a professional real estate news analyst.

        Your task is to answer the user's question using ONLY the provided article context.
        Do NOT use external knowledge or make assumptions.

        If the answer cannot be found in the article context, clearly say:
        "I don't know based on the provided article."

        Conversation summary (for continuity only):
        {summary}

        Recent conversation (for reference only):
        {chat_history}

        Article context (primary source of truth):
        {context}

        User question:
        {query}

        Guidelines:
        - Be factual and precise.
        - Cite names, companies, locations, and dates exactly as mentioned.
        - Keep the answer concise and well-structured.
        - Do not hallucinate or speculate.
        - If multiple viewpoints are mentioned, summarize them neutrally.
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

    recent_msgs.append(f"User: {query}")
    recent_msgs.append(f"Assistant: {answer}")

    if len(recent_msgs) > SUMMARY_TRIGGER:
        summary = summarize_conversation(recent_msgs)
        recent_msgs[:] = recent_msgs[-MAX_RECENT_MESSAGES * 2 :]

    return answer, summary, recent_msgs

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
