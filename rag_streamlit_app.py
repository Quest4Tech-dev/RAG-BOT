import os
import streamlit as st
import faiss
import numpy as np
from dotenv import load_dotenv
# from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

st.set_page_config(page_title="RAG QA Bot", layout="wide")
st.title("📄 Retrieval-Augmented Generation (RAG) QA Bot")

pdf_path = "data/Campus_Trade_Project.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

texts = []
vectors = []

for doc in docs:
    texts.append(doc.page_content)
    response = genai.embed_content(
        model="models/embedding-001",
        content=doc.page_content,
        task_type="retrieval_document"
    )
    vectors.append(response["embedding"])

dimension = len(vectors[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(vectors).astype("float32"))

query = st.text_input("Ask a question based on the document:")

if query:
    query_vector = genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="retrieval_query"
    )["embedding"]

    D, I = index.search(np.array([query_vector]).astype("float32"), k=3)
    relevant_chunks = [texts[i] for i in I[0]]

    context = "\n".join(relevant_chunks)
    prompt = f"Answer the following question based on the context below:\n\nContext:\n{context}\n\nQuestion:\n{query}"

    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
    response = model.generate_content(prompt)
    
    st.subheader("📌 Answer:")
    st.write(response.text)
