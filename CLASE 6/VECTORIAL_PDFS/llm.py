"""
llm.py

Generación de respuestas usando Gemini.
"""

from langchain_google_genai import ChatGoogleGenerativeAI

from config import (
    GOOGLE_API_KEY,
    CHAT_MODEL,
)

llm = ChatGoogleGenerativeAI(
    model=CHAT_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
)


def generate_answer(question, documents):

    context = ""

    for i, doc in enumerate(documents, start=1):

        context += (
            f"\nDocumento {i}\n"
            f"Fuente: {doc['source']}\n"
            f"{doc['content']}\n"
        )

    prompt = f"""
Sos un asistente RAG.

Respondé utilizando únicamente la información
del contexto.

Si la respuesta no aparece en el contexto,
decí explícitamente que no encontraste información.

Contexto:

{context}

Pregunta:

{question}

Respuesta:
"""

    response = llm.invoke(prompt)

    return response.content