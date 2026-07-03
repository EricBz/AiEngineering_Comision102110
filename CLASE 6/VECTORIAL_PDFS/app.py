"""
app.py

RAG con:
- ChromaDB
- Gemini
- Ranking
- ReRanking
"""

from retriever import retrieve_documents
from llm import generate_answer


def print_documents(documents):
    print("\n========== DOCUMENTOS RECUPERADOS ==========\n")

    for i, doc in enumerate(documents, start=1):
        print(f"Documento #{i}")
        print(f"Score: {doc['score']:.4f}")
        print(f"Archivo: {doc['source']}")
        print("-" * 50)
        print(doc["content"])
        print("-" * 50)
        print()


def chat():

    print("=" * 60)
    print("RAG con Chroma + Gemini + ReRanking")
    print("Escribí 'salir' para terminar.")
    print("=" * 60)

    while True:

        question = input("\nPregunta: ")

        if question.lower() in ["salir", "exit", "quit"]:
            break

        print("\nBuscando información...")

        retrieved_docs = retrieve_documents(question)

        if len(retrieved_docs) == 0:
            print("No se encontraron documentos.")
            continue

        print_documents(retrieved_docs)

        answer = generate_answer(
            question=question,
            documents=retrieved_docs
        )

        print("\n================ RESPUESTA ================\n")
        print(answer)
        print("\n===========================================")


if __name__ == "__main__":
    chat()