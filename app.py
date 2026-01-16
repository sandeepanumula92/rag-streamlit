import streamlit as st
import os
import tempfile
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# THIS IS THE CRITICAL CHANGE FOR 2026
from langchain_classic.chains import RetrievalQA

st.set_page_config(page_title="Gemini RAG Tutor", layout="wide")
st.title("📚 Gemini + Pinecone RAG (2026 Edition)")

# --- 1. Sidebar for API Keys ---
with st.sidebar:
    st.header("Setup")
    google_api_key = st.text_input("Google API Key", type="password")
    pinecone_api_key = st.text_input("Pinecone API Key", type="password")
    index_name = st.text_input("Pinecone Index Name", value="rag-index")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# --- 2. Logic ---
if uploaded_file and google_api_key and pinecone_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key
    os.environ["PINECONE_API_KEY"] = pinecone_api_key

    # Save and Load PDF
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(uploaded_file.getbuffer())
        file_path = tf.name

    with st.status("Processing Document...", expanded=True) as status:
        st.write("Reading PDF...")
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        st.write("Splitting into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)

        st.write("Generating Gemini Embeddings...")
        # Using Google embeddings (768 dims) to avoid Python 3.13 HF issues
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        
        st.write("Updating Pinecone...")
        vectorstore = PineconeVectorStore.from_documents(
            splits, embeddings, index_name=index_name
        )
        status.update(label="Ready! Ask your questions below.", state="complete")

    # --- 3. Chat Interface ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask about your PDF"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # Setup RAG Chain using the Classic import
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=vectorstore.as_retriever()
        )

        with st.spinner("Gemini is thinking..."):
            response = qa_chain.invoke(prompt)
            answer = response["result"]
            
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
else:
    st.info("Please enter your Google & Pinecone keys and upload a PDF.")
