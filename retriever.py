from embedder import load_faiss_vector_store
from langchain.schema import Document

def retrieve_similar_documents(query, k=3, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    # Load the vector store
    vectorstore = load_faiss_vector_store(model_name, load_path="data")
    
    # Search for similar documents
    docs = vectorstore.similarity_search(query, k=k)
    
    print(f"\n🔎 Top {k} Results for: \"{query}\"\n")
    for i, doc in enumerate(docs):
        print(f"[{i+1}] {doc.metadata.get('source', 'Unknown')}")
        print(doc.page_content[:250] + "...\n")
    return docs

def retrieve_with_mmr(query, k=3, fetch_k=10, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    vectorstore = load_faiss_vector_store(model_name, load_path="data")
    
    docs = vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)
    
    print(f"\n🎯 MMR Top {k} Results for: \"{query}\"\n")
    for i, doc in enumerate(docs):
        print(f"[{i+1}] {doc.metadata.get('source', 'Unknown')}")
        print(doc.page_content[:250] + "...\n")
    return docs
