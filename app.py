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
st.set_page_config(page_title="RAG vs LLM", layout="wide")
st.title("📚 RAG vs. Standalone LLM Comparison")

# --- 1. Load Secrets Automatically ---
# Make sure you have added these to the 'Secrets' tab in Streamlit Cloud Settings!
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    pinecone_api_key = st.secrets["PINECONE_API_KEY"]
except KeyError:
    st.error("Missing API Keys! Please add GOOGLE_API_KEY and PINECONE_API_KEY to your Streamlit Secrets.")
    st.stop()

# --- 2. Sidebar Configuration ---
with st.sidebar:
    st.header("Configuration")
    # Using a unique key="pdf_uploader" prevents the DuplicateElementId error
    uploaded_file = st.file_uploader("Upload your Expert PDF", type="pdf", key="pdf_uploader")
    
    index_name = st.text_input("Pinecone Index Name", value="rag-index")
    
    st.divider()
    st.header("Intelligence Mode")
    mode = st.radio(
        "Choose Answer Mode:",
        ["RAG (Uses your PDF)", "Standalone LLM (General Knowledge)"],
        index=0
    )

# --- 3. Processing & Caching ---
# We cache the vectorstore so it doesn't re-upload to Pinecone every time you chat
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
    vector_db = PineconeVectorStore.from_documents(
        splits, embeddings, index_name=i_name
    )
    return vector_db

# --- 4. Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Initialize the LLM (Gemini)
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=google_api_key,
        safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
    )

    with st.spinner(f"Generating {mode} answer..."):
        if mode == "RAG (Uses your PDF)" and uploaded_file:
            # RAG Path
            v_store = process_pdf(uploaded_file.getvalue(), google_api_key, pinecone_api_key, index_name)
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm, chain_type="stuff", retriever=v_store.as_retriever()
            )
            response = qa_chain.invoke(prompt)["result"]
        else:
            # Standalone Path (or if no file uploaded)
            if mode == "RAG (Uses your PDF)" and not uploaded_file:
                response = "Please upload a PDF first to use RAG mode!"
            else:
                response = llm.invoke(prompt).content

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
