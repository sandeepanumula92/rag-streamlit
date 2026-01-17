import streamlit as st
import os
import tempfile
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_google_genai import HarmCategory, HarmBlockThreshold

# --- Page Config ---
st.set_page_config(page_title="RAG Comparison", layout="wide")
st.title("📚 RAG vs. Standalone LLM")

# --- 1. Load Secrets Automatically ---
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    pinecone_api_key = st.secrets["PINECONE_API_KEY"]
    # Index name is now pulled from secrets, not the UI
    index_name = st.secrets["PINECONE_INDEX_NAME"]
except KeyError as e:
    st.error(f"Missing Secret: {e}. Please add it to your Streamlit Secrets.")
    st.stop()

# --- 2. Sidebar (Now much cleaner) ---
with st.sidebar:
    st.header("Document Management")
    uploaded_file = st.file_uploader("Upload Expert PDF", type="pdf", key="pdf_uploader")
    
    st.divider()
    st.header("Mode")
    mode = st.radio(
        "Answer Mode:",
        ["RAG (Uses PDF)", "Standalone (General)"],
        index=0
    )

# --- 3. Processing Logic (Cached) ---
@st.cache_resource
def process_pdf(file_content, g_key, p_key, i_name):
    os.environ["GOOGLE_API_KEY"] = g_key
    os.environ["PINECONE_API_KEY"] = p_key
    
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(file_content)
        file_path = tf.name

    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    return PineconeVectorStore.from_documents(splits, embeddings, index_name=i_name)

# --- 4. Chat Interface ---
if prompt := st.chat_input("Ask a question..."):
    st.chat_message("user").write(prompt)

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=google_api_key,
        safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
    )

    with st.spinner(f"Using {mode}..."):
        if mode == "RAG (Uses PDF)" and uploaded_file:
            v_store = process_pdf(uploaded_file.getvalue(), google_api_key, pinecone_api_key, index_name)
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm, chain_type="stuff", retriever=v_store.as_retriever()
            )
            response = qa_chain.invoke(prompt)["result"]
        else:
            response = llm.invoke(prompt).content if mode == "Standalone (General)" else "Please upload a PDF for RAG mode."

    st.chat_message("assistant").write(response)
