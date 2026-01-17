import streamlit as st
import os
import tempfile
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_google_genai import HarmCategory, HarmBlockThreshold

st.set_page_config(page_title="RAG vs LLM Comparison", layout="wide")
st.title("📚 Intelligence Comparison: RAG vs. Standalone LLM")

# --- 1. Load Secrets Automatically ---
# Streamlit will look for these in the "Secrets" dashboard (Cloud)
# or in .streamlit/secrets.toml (Local)
google_api_key = st.secrets["GOOGLE_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
# Note: We keep the PDF uploader and index name in the sidebar
with st.sidebar:
    st.header("Configuration")
    index_name = st.text_input("Pinecone Index Name", value="rag-index")
    uploaded_file = st.file_uploader("Upload your Expert PDF", type="pdf")
    
    st.header("2. Knowledge Source")
    uploaded_file = st.file_uploader("Upload your Expert PDF", type="pdf")
    
    st.header("3. Intelligence Mode")
    # THE TOGGLE: This determines our logic below
    mode = st.radio(
        "Choose how the AI answers:",
        ["RAG (Uses your PDF)", "Standalone LLM (General Knowledge)"]
    )

# --- 2. Processing Layer ---
if uploaded_file and google_api_key and pinecone_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key
    os.environ["PINECONE_API_KEY"] = pinecone_api_key

    # Temporary storage and PDF processing
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(uploaded_file.getbuffer())
        file_path = tf.name

    # We wrap the indexing in st.cache_resource so it doesn't re-run every time you chat
    @st.cache_resource
    def get_vectorstore(path):
        loader = PyPDFLoader(path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        return PineconeVectorStore.from_documents(splits, embeddings, index_name="rag-index")

    vectorstore = get_vectorstore(file_path)

    # --- 3. Interaction Layer ---
    if prompt := st.chat_input("Ask a specific question about your document"):
        st.chat_message("user").write(prompt)
        
        # Initialize the base Brain (Gemini)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
        )

        with st.spinner(f"Thinking in {mode} mode..."):
            if mode == "RAG (Uses your PDF)":
                # --- RAG PATH ---
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
                )
                response = qa_chain.invoke(prompt)["result"]
                st.chat_message("assistant").write(f"**[RAG ANSWER]:** {response}")
            else:
                # --- NON-RAG PATH ---
                # We skip Pinecone and just ask the LLM directly
                response = llm.invoke(prompt).content
                st.chat_message("assistant").write(f"**[GENERAL LLM ANSWER]:** {response}")
else:
    st.info("Please provide your keys and upload a PDF to unlock the RAG mode.")
