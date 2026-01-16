import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains import RetrievalQA
import tempfile

st.set_page_config(page_title="Free RAG Tutor", layout="wide")
st.title("📚 Simple RAG Assistant")

# --- Setup Sidebar for API Keys ---
with st.sidebar:
    st.header("Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password")
    pinecone_api_key = st.text_input("Pinecone API Key", type="password")
    pinecone_index = st.text_input("Pinecone Index Name", value="rag-index")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# --- Core RAG Logic ---
if uploaded_file and groq_api_key and pinecone_api_key:
    # Save uploaded file temporarily to load it
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(uploaded_file.getbuffer())
        file_path = tf.name

    # 1. Process Document
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    # 2. Initialize Vector DB
    os.environ["PINECONE_API_KEY"] = pinecone_api_key
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = PineconeVectorStore.from_documents(
        splits, embeddings, index_name=pinecone_index
    )
    
    # 3. Setup QA Chain
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
    )

    st.success("Document processed and indexed!")

    # --- Chat Interface ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask something about your PDF"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.spinner("Thinking..."):
            response = qa_chain.invoke(prompt)
            answer = response["result"]
            
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
else:
    st.info("Please upload a PDF and enter your API keys in the sidebar to begin.")
