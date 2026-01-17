import streamlit as st
import os
import tempfile
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA

# NEW: Import for safety settings
from langchain_google_genai import HarmCategory, HarmBlockThreshold

st.set_page_config(page_title="2026 Gemini RAG", layout="wide")
st.title("📚 Gemini + Pinecone RAG")

with st.sidebar:
    st.header("Setup")
    google_api_key = st.text_input("Google API Key", type="password")
    pinecone_api_key = st.text_input("Pinecone API Key", type="password")
    index_name = st.text_input("Pinecone Index Name", value="rag-index")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and google_api_key and pinecone_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key
    os.environ["PINECONE_API_KEY"] = pinecone_api_key

    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(uploaded_file.getbuffer())
        file_path = tf.name

    with st.status("Indexing Document...", expanded=False):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)

        # Gemini 2026 standard embedding model
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        
        vectorstore = PineconeVectorStore.from_documents(
            splits, embeddings, index_name=index_name
        )

    # --- Chat Interface ---
    if prompt := st.chat_input("Ask about your PDF"):
        st.chat_message("user").write(prompt)
        
        # Setup LLM with SAFETY SETTINGS to prevent the ChatGoogleGenerativeAIError
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", # Updated for 2026
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=vectorstore.as_retriever()
        )

        try:
            with st.spinner("Gemini is analyzing..."):
                response = qa_chain.invoke(prompt)
                st.chat_message("assistant").write(response["result"])
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Check if your API Key is valid and if the content violates safety guidelines.")
