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
st.set_page_config(page_title="RAG Comparison 2026", layout="wide")
st.title("📚 RAG vs. Standalone LLM")

# --- 1. Load Secrets ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]
except KeyError as e:
    st.error(f"Missing Secret: {e}. Please add it to Streamlit Cloud Secrets.")
    st.stop()

# --- 2. Sidebar ---
with st.sidebar:
    st.header("Upload")
    uploaded_file = st.file_uploader("Upload Expert PDF", type="pdf", key="pdf_uploader")
    st.divider()
    st.header("Mode")
    mode = st.radio("Answer Mode:", ["RAG (Uses PDF)", "Standalone (General)"], index=0)

# --- 3. Processing (Cached) ---
@st.cache_resource
def get_vector_db(file_bytes, g_key, p_key, i_name):
    os.environ["GOOGLE_API_KEY"] = g_key
    os.environ["PINECONE_API_KEY"] = p_key
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
        tf.write(file_bytes)
        file_path = tf.name

    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # This will now give a clear error if dimensions don't match
    try:
        return PineconeVectorStore.from_documents(splits, embeddings, index_name=i_name)
    except Exception as e:
        st.error(f"Pinecone Error: Ensure your index '{i_name}' has 768 dimensions.")
        st.stop()

# --- 4. Chat logic ---
if prompt := st.chat_input("Ask a question..."):
    st.chat_message("user").write(prompt)

    # Use 1.5 Flash for the LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2, # Lower temperature = more factual
        safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
    )

    try:
        with st.spinner(f"Running {mode}..."):
            if mode == "RAG (Uses PDF)" and uploaded_file:
                v_db = get_vector_db(uploaded_file.getvalue(), GOOGLE_API_KEY, PINECONE_API_KEY, INDEX_NAME)
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm, chain_type="stuff", retriever=v_db.as_retriever()
                )
                response = qa_chain.invoke(prompt)["result"]
            else:
                if mode == "RAG (Uses PDF)" and not uploaded_file:
                    response = "⚠️ Please upload a PDF first to use RAG mode."
                else:
                    response = llm.invoke(prompt).content
            
            st.chat_message("assistant").write(response)

    except Exception as e:
        # This catches the ChatGoogleGenerativeAIError and explains it
        st.error("⚠️ The Google API rejected the request.")
        if "429" in str(e):
            st.warning("Rate limit reached. Please wait 60 seconds.")
        elif "400" in str(e):
            st.info("Check if your Pinecone Index dimensions are set to 768.")
        else:
            st.write(f"Details: {e}")
