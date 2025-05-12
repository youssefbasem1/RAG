from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from pathlib import Path

def load_documents(folder_path="documents"):
    docs = []
    for file_path in Path(folder_path).rglob("*"):
        try:
            if file_path.suffix == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif file_path.suffix == ".docx":
                loader = Docx2txtLoader(str(file_path))
            elif file_path.suffix == ".txt":
                loader = TextLoader(str(file_path))
            else:
                print(f"Unsupported format: {file_path}")
                continue
            docs.extend(loader.load())
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
    return docs
