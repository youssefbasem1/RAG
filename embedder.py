from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def create_faiss_vector_store(chunks, model_name, save_path="data"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(save_path)
    return vectorstore

def load_faiss_vector_store(model_name, load_path="data"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)
