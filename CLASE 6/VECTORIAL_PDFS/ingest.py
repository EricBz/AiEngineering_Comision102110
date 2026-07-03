"""
ingest.py

Lee todos los PDFs de /docs, los divide en chunks,
genera embeddings con Gemini y los almacena
en ChromaDB de forma persistente.

Actualizado para 2026.
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from config import (
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
    CHROMA_PATH,
    DOCS_PATH,
    COLLECTION_NAME
)


def load_documents():

    documents = []

    pdfs = Path(DOCS_PATH).glob("*.pdf")

    for pdf in pdfs:

        print(f"Cargando {pdf.name}")

        loader = PyPDFLoader(str(pdf))
        docs = loader.load()

        documents.extend(docs)

    return documents


def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    return splitter.split_documents(documents)


def build_embeddings():

    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY
    )


def create_vector_db(chunks):

    embeddings = build_embeddings()

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME
    )

    return vectordb


def main():

    print("=" * 60)
    print("INDEXANDO PDFs")
    print("=" * 60)

    documents = load_documents()

    print(f"\nDocumentos cargados: {len(documents)}")

    chunks = split_documents(documents)

    print(f"Chunks generados: {len(chunks)}")

    create_vector_db(chunks)

    print("\nBase vectorial creada correctamente.")


if __name__ == "__main__":
    main()