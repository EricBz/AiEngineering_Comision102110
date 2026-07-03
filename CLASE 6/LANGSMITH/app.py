from dotenv import load_dotenv
# Forzamos la carga del archivo .env antes de cualquier otra cosa
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

# Ahora sí inicializamos el modelo de manera segura
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

def generar_respuesta(pregunta: str) -> str:
    respuesta = llm.invoke(pregunta)
    return respuesta.content
