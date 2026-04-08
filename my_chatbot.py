import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.chat_models import ChatOllama

st.header("My Chatbot")

with st.sidebar:
    file = st.file_uploader("Upload PDF", type="pdf")

if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = text_splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(chunks, embeddings)

    user_question = st.text_input("Ask question")

    if user_question:
        llm = ChatOllama(model="llama3")

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vector_store.as_retriever()
        )

        response = qa.run(user_question)
        st.write(response)