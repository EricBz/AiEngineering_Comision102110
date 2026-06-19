from data.knowledge_base import DOCUMENTS

def search_documents(query: str):
    query = query.lower()

    results = []

    for doc in DOCUMENTS:
        if any(word in doc["content"].lower() for word in query.split()):
            results.append(doc["content"])

    return results