"""
chroma.py

Inicializa y devuelve la instancia de ChromaDB.
"""

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import (
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
    CHROMA_PATH,
    COLLECTION_NAME,
)


def get_embeddings():

    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )


def get_vectordb():

    embeddings = get_embeddings()

    return Chroma(
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )