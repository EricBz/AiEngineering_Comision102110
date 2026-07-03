from dotenv import load_dotenv
load_dotenv() # Carga las llaves del .env de forma local

from langchain_google_genai import ChatGoogleGenerativeAI

# Tu configuración real de producción para Coderhouse
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def generar_respuesta(pregunta: str) -> str:
    """Función principal de la app que procesa las consultas."""
    respuesta = llm.invoke(pregunta)
    return respuesta.content
