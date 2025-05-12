from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from retriever import retrieve_similar_documents
from embedder import load_faiss_vector_store
import os

def get_rag_prompt_template():
    template = """
You are an expert assistant. Use the provided context to answer the question as accurately as possible.
If the answer cannot be found in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
    return PromptTemplate(
        input_variables=["context", "question"],
        template=template.strip()
    )

def get_summary_prompt_template():
    template = """
You are an expert summarizer. Use the context below to generate a concise, accurate summary.

Context:
{context}

Summary:
"""
    return PromptTemplate(
        input_variables=["context"],
        template=template.strip()
    )

def get_llm():
    return ChatOpenAI(
        temperature=0.2,
        openai_api_key=os.getenv("GROQ_API_KEY"),  # Needed by LangChain
        openai_api_base=os.getenv("GROQ_BASE_URL"),
        model=os.getenv("GROQ_MODEL")
    )

def build_rag_chain(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    vectorstore = load_faiss_vector_store(model_name, load_path="data")
    retriever = vectorstore.as_retriever()
    llm = get_llm()
    prompt = get_rag_prompt_template()

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",  # or "map_reduce", "refine"
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
