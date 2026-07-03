from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DB_PATH = "./chroma_db"

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

vector_store = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)


def search_documents(query: str, k: int = 3):
    """
    Devuelve una lista de strings con los documentos
    más relevantes para la consulta.
    """

    docs = vector_store.similarity_search(
        query=query,
        k=k
    )

    return [doc.page_content for doc in docs]