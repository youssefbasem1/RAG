from document_loader import load_documents
from splitter import split_documents
from embedder import create_faiss_vector_store, load_faiss_vector_store
from retriever import retrieve_similar_documents, retrieve_with_mmr
from rag_chain import build_rag_chain
from evaluator import evaluate_retrieval
from dotenv import load_dotenv
load_dotenv()

def build_vectorstore_pipeline():
    print("🔍 Loading documents...")
    documents = load_documents("documents")

    print("✂️ Splitting documents...")
    chunks = split_documents(documents, chunk_size=500, chunk_overlap=50)

    print("🧠 Embedding and saving to FAISS...")
    create_faiss_vector_store(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2")


def run_rag_query(query):
    print(f"\n💬 Running RAG for query: \"{query}\"")
    rag = build_rag_chain()
    result = rag(query)

    print("\n🧠 Final Answer:\n", result["result"])
    print("\n📄 Source Documents:")
    for doc in result["source_documents"]:
        print("-", doc.metadata.get("source", "Unknown"))


def test_rag_retrieval_configs(query, gold_sources):
    print("\n🔬 Comparing retrieval configurations...\n")

    # Strategy 1: Similarity Search
    print("🔹 Similarity Search:")
    docs_sim = retrieve_similar_documents(query, k=3)
    sources_sim = [doc.metadata.get("source", "") for doc in docs_sim]
    eval_sim = evaluate_retrieval(gold_sources, sources_sim)
    print("Eval:", eval_sim)

    # Strategy 2: MMR
    print("\n🔹 MMR Search:")
    docs_mmr = retrieve_with_mmr(query, k=3)
    sources_mmr = [doc.metadata.get("source", "") for doc in docs_mmr]
    eval_mmr = evaluate_retrieval(gold_sources, sources_mmr)
    print("Eval:", eval_mmr)


def test_multiple_embeddings(query, gold_sources):
    models = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2"
    ]

    for model in models:
        print(f"\n🔁 Testing with embedding model: {model}")
        documents = load_documents("documents")
        chunks = split_documents(documents)

        save_path = f"data_{model.split('/')[-1]}"
        create_faiss_vector_store(chunks, model_name=model, save_path=save_path)
        vectorstore = load_faiss_vector_store(model_name=model, load_path=save_path)

        docs = vectorstore.similarity_search(query, k=3)
        retrieved_sources = [doc.metadata.get("source", "") for doc in docs]
        print("Eval:", evaluate_retrieval(gold_sources, retrieved_sources))

if __name__ == "__main__":
    build_vectorstore_pipeline()
    
    query = "What are transformers?"
    gold_sources = ["what_are_transformers.txt"]

    run_rag_query(query)
    test_rag_retrieval_configs(query, gold_sources)
    test_multiple_embeddings(query, gold_sources)
