import os
from bson import ObjectId
from datetime import datetime
from pymongo import MongoClient
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import JSONLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from huggingface_hub import  InferenceClient

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN2")

if not MONGO_URI or not HUGGINGFACE_TOKEN:
    raise ValueError("Missing environment variables: MONGO_URI or HUGGINGFACE_TOKEN")

DB_NAME = "blogging"
PERSIST_DIR = "./chroma_langchain_db"

model_name = "Qwen/Qwen2.5-7B-Instruct"
llm = InferenceClient(model=model_name,token=HUGGINGFACE_TOKEN)

def export_mongodb_to_json(file_path="data/blogging.json"):
    client = MongoClient(MONGO_URI)
    db = client["blogging"]

    blogs_data = list(db["blogs"].find({}))
    users_data = list(db["users"].find({}))

    for blog in blogs_data:
        for key, value in blog.items():
            if isinstance(value, ObjectId):
                blog[key] = str(value)
            elif isinstance(value, datetime):
                blog[key] = value.isoformat()

    for user in users_data:
        for key, value in user.items():
            if isinstance(value, ObjectId):
                user[key] = str(value)
            elif isinstance(value, datetime):
                user[key] = value.isoformat()

    export_data = {
        "blogs": blogs_data,
        "users": users_data
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=4)

    return file_path

def load_json_documents(file_path):
    loader = JSONLoader(
        file_path=file_path,
        jq_schema=".users[]",
        text_content=False
    )

    docs = loader.load()
    print(f"Loaded {len(docs)} documents from JSON")
    return docs

def split_documents(docs):
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000,chunk_overlap=300,separators=["\n}\n", "\n]\n", "\n"])
    all_splits = text_splitter.split_documents(docs)
    print(f"Total chunks created: {len(all_splits)}")
    return all_splits

def init_vector_store(chunks):
    print("Initializing embeddings...")
    embedding = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")

    print("Initializing Chroma vector store...")
    vector_store = Chroma(
        collection_name="blog_vector_collection",
        embedding_function=embedding,
        persist_directory=PERSIST_DIR
    )

    if not vector_store.get()["ids"]:
        print("Adding documents to ChromaDB...")
        vector_store.add_documents(chunks)
        print("Documents stored in ChromaDB")
    else:
        print("Using existing ChromaDB vector store")

    return vector_store

def answer_query(vector_store, query: str, k: int = 3):
    print(f"\nSearching for relevant chunks for query: {query}")
    retrieved_context = vector_store.similarity_search(query, k=k)

    if not retrieved_context:
        return "No relevant information found in the knowledge base."

    context = "\n\n".join([doc.page_content for doc in retrieved_context])

    system_prompt = (
        "You are an expert data assistant. "
        "You are given user data in JSON format. "
        "You must ONLY answer based on this JSON data, no SQL, no assumptions. "
        "If the query is 'give me all users', output a list of all user records from context. "
        "If you cannot find the data, reply exactly: 'I don't know based on the given data'."
    )

    response = llm.chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        max_tokens=800
    )
    return response.choices[0].message.content


def main():
    file_path = export_mongodb_to_json()
    docs = load_json_documents(file_path)
    chunks = split_documents(docs)
    vector_store = init_vector_store(chunks)

    print("\nWelcome to the MongoDB RAG Assistant. Type 'exit' to quit.")
    while True:
        user_query = input("\nEnter your question: ")
        if user_query.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
        try:
            answer = answer_query(vector_store, user_query)
            print("\nResponse:\n", answer)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
