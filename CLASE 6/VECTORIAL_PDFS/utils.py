"""
utils.py

Funciones auxiliares para impresión y formateo.
"""


def separator(title: str = ""):

    print("\n" + "=" * 70)

    if title:
        print(title)

    print("=" * 70)


def print_documents(documents):

    separator("DOCUMENTOS RECUPERADOS")

    for i, doc in enumerate(documents, start=1):

        print(f"\nDocumento {i}")
        print(f"Fuente: {doc['source']}")
        print(f"Score: {doc['score']:.4f}")

        print("-" * 60)
        print(doc["content"])
        print("-" * 60)


def build_context(documents):

    context = ""

    for i, doc in enumerate(documents, start=1):

        context += (
            f"\nDocumento {i}\n"
            f"Fuente: {doc['source']}\n"
            f"{doc['content']}\n"
        )

    return context