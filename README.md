# Retrieval-Augmented Generation (RAG)

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system using open-source tools. It retrieves relevant documents and generates grounded answers using the **Groq LLaMA3-70B model**.

---

##  Setup Instructions

1. **Clone the repo and create a virtual environment:**

```bash
git clone <your_repo_url>
cd RAG_Assignment
python3 -m venv ragenv
source ragenv/bin/activate
pip install -r requirements.txt
```

2. **Set up `.env`:**

```
GROQ_API_KEY=your_groq_key_here
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=llama3-70b-8192
```

3. **Add documents:**

Place your `.pdf`, `.txt`, and `.docx` files inside the `documents/` folder.

4. **Run the pipeline:**

```bash
python main.py
```

---

## RAG System Architecture

| Component            | Tool / Library                                                   |
| -------------------- | ---------------------------------------------------------------- |
| **Document Loaders** | LangChain + langchain\_community                                 |
| **Chunking**         | `RecursiveCharacterTextSplitter`                                 |
| **Embeddings**       | `sentence-transformers/all-MiniLM-L6-v2` and `all-mpnet-base-v2` |
| **Vector Store**     | FAISS (stored locally in `data/`)                                |
| **Retriever**        | Similarity Search + MMR                                          |
| **LLM**              | Groq LLaMA3-70B via OpenAI-compatible LangChain interface        |
| **Evaluation**       | Precision / Recall / F1 on retrieval                             |
| **Prompting**        | Q\&A and Summarization prompt templates                          |

---

## Retrieval Strategy Results

**Query:** `"What are transformers?"`

| Strategy   | Top Result Document         | Precision | Recall | F1  |
| ---------- | --------------------------- | --------- | ------ | --- |
| Similarity | what\_are\_transformers.txt | 0.5       | 0.5    | 0.5 |
| MMR        | what\_are\_transformers.txt | 0.5       | 0.5    | 0.5 |

---

### Embedding Model Comparison

| Embedding Model   | Precision | Recall | F1  |
| ----------------- | --------- | ------ | --- |
| all-MiniLM-L6-v2  | 0.5       | 0.5    | 0.5 |
| all-mpnet-base-v2 | 1.0       | 1.0    | 1.0 |

---

### Prompt Template Variants

* **Q\&A Prompt**: Uses context to answer a direct question
* **Summarization Prompt**: Generates a concise summary of the context

Both templates tested successfully with the Groq LLaMA3-70B model.

---

## Evaluation Metrics (after path normalization)

* **Precision**: Correct retrieved documents / Total retrieved
* **Recall**: Correct retrieved / Total relevant
* **F1 Score**: Harmonic mean of precision and recall

Used exact file names (normalized with `os.path.basename`) to match retrieved documents with ground truth.

---

## Strengths

* Fully modular and readable Python code
* Accurate Groq LLM generation via API
* Working FAISS-based vector store
* Easy to evaluate multiple retrieval strategies
* Uses open-source embeddings for portability

---

## Weaknesses

* FAISS similarity may retrieve off-topic results if documents are loosely relevant
* RAG system relies heavily on chunk quality (context boundaries)

---

## Challenges and Solutions

| Challenge                           | Solution                                       |
| ----------------------------------- | ---------------------------------------------- |
| FAISS `.pkl` security restriction   | Set `allow_dangerous_deserialization=True`     |
| Path mismatch during evaluation     | Normalized paths with `os.path.basename()`     |
| `ChatOpenAI` needs `OPENAI_API_KEY` | Mapped `GROQ_API_KEY` to OpenAI-compatible var |
| Deprecation warnings in LangChain   | Updated to `langchain-community` imports       |

---

## Document Corpus

Documents used for testing:

* `what_are_transformers.txt`
* `vision_transformers_at_scale.pdf`
* `alphafold2_protein_prediction.pdf`

You may add your own files in the `documents/` folder.

---

## Configuration & Saved Files

* `data/` – contains saved FAISS index (one per embedding model)
* `.env` – contains API key settings (not pushed to GitHub)
* All models/embeddings are downloaded dynamically and cached by `sentence-transformers`

---

## Acknowledgements

* LangChain, HuggingFace, FAISS, Groq API
* “Attention is All You Need” paper by Vaswani et al.
