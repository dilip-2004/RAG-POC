import os
import shutil
import pymupdf4llm
from langchain.text_splitter import MarkdownTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")

def create_or_load_faiss(pdf_files):
    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")

    splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)

    all_chunks = []
    for pdf_file in pdf_files:
        md_test = pymupdf4llm.to_markdown(pdf_file)
        chunks = splitter.create_documents([md_test])
        all_chunks.extend(chunks)

    vector_store = FAISS.from_documents(all_chunks, embedding_model)
    vector_store.save_local("faiss_index")
    
    return vector_store

def retrieve(vector_store,query):
    retrieved_docs = vector_store.similarity_search(query, k=8)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    return context