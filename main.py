import streamlit as st
from huggingface_hub import InferenceClient
from langchain_helper import create_or_load_faiss,retrieve
import os
from dotenv import load_dotenv

load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

model_name = "Qwen/Qwen2.5-7B-Instruct"
huggingface_token = HUGGINGFACE_TOKEN
client = InferenceClient(model=model_name,token=huggingface_token)

st.title("PDF RAG System")
st.write("upload your PDF and ask Questions")

uploaded_files = st.file_uploader("", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    pdf_path = []
    os.makedirs("uploaded_pdfs", exist_ok=True)

    for uploaded_file in uploaded_files:
        file_path = os.path.join("uploaded_pdfs",uploaded_file.name)
        with open(file_path,"wb") as f:
            f.write(uploaded_file.getbuffer())
        pdf_path.append(file_path)

    st.success(f"Uploaded {len(pdf_path)} PDFs.")

    vector_store = create_or_load_faiss(pdf_path)

    query = st.text_input("Ask a questions about your PDFs")
    if query:
        context = retrieve(vector_store,query)

        response = client.chat_completion(
            messages=[
                {"role": "system", "content": f"You are a helpful assistant. First, use the following PDF context to answer the question. If the context does not contain the answer, then use your own knowledge.\n\nContext:\n{context}"},
                {"role": "user", "content": query}
            ],
            max_tokens=800,
            temperature=0.3
        )

        st.subheader("Response")
        st.write(response.choices[0].message.content)