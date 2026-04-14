import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# Load env
load_dotenv()

st.set_page_config(page_title="RAG App", layout="wide")
st.title("📄 RAG Document Q&A")

# ---------------------------
# Upload PDF
# ---------------------------
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("File uploaded successfully!")

    # ---------------------------
    # Load & Split Document
    # ---------------------------
    loader = PyPDFLoader(file_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(data)

    st.info(f"Total chunks created: {len(chunks)}")

    # ---------------------------
    # Create Vector Store
    # ---------------------------
    embeddings = MistralAIEmbeddings()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="vector_db"
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3},
        search_type="mmr"
    )

    # ---------------------------
    # Query Input
    # ---------------------------
    query = st.text_input("Ask a question from the document:")

    if query:
        with st.spinner("Thinking..."):

            retrieved_docs = retriever.invoke(query)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])

            llm = ChatOllama(
                model="gemma4:31b-cloud",
                temperature=0.2,
                base_url="https://ollama.com",
                api_key=os.getenv("OLLAMA_API_KEY")
            )

            prompt_template = ChatPromptTemplate(
                [
                    ("system", "You are a helpful assistant that answers questions based on the provided context which is retrieved from user provided document so your answer must be in that context. if given questions answer is not present in context means document then say : 'Sorry, I don't have that information.'"),
                    ("human", "{context}\n\nQuestion: {question}")
                ]
            )

            prompt = prompt_template.invoke(
                {
                    "context": context,
                    "question": query
                }
            )

            response = llm.invoke(prompt)

            # ---------------------------
            # Output
            # ---------------------------
            st.subheader("Answer:")
            st.write(response.content)

            # Debug (optional)
            with st.expander("📚 Retrieved Context"):
                st.write(context)