from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DB_PATH = "./chroma_db"
TXT_PATH = "./data/knowledge.txt"

def build_database():

    if Path(DB_PATH).exists() and any(Path(DB_PATH).iterdir()):
        print("Base vectorial ya existente.")
        return

    print("Construyendo base vectorial...")

    with open(TXT_PATH, "r", encoding="utf8") as file:
        text = file.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    documents = [
        Document(
            page_content=chunk,
            metadata={"source": "knowledge.txt"}
        )
        for chunk in chunks
    ]

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    print(f"Base creada con {len(documents)} fragmentos.")


if __name__ == "__main__":
    build_database()