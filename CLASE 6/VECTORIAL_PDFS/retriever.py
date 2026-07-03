from sentence_transformers import CrossEncoder
from chroma import get_vectordb

vectordb = get_vectordb()

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

def retrieve_documents(question: str):

    docs = vectordb.similarity_search_with_score(
        question,
        k=20
    )

    pairs = []

    for doc, _ in docs:
        pairs.append([question, doc.page_content])

    rerank_scores = reranker.predict(pairs)

    ranked = []

    for (doc, vector_score), rerank_score in zip(docs, rerank_scores):

        ranked.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Desconocido"),
            "vector_score": float(vector_score),
            "rerank_score": float(rerank_score),
            "score": float(rerank_score),
        })

    ranked.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return ranked[:5]