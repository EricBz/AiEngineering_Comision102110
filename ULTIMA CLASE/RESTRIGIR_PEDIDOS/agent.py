import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableBranch, RunnablePassthrough

# Carga las variables desde el archivo .env al entorno del sistema
load_dotenv()

# Instanciamos los modelos (detectan automáticamente GOOGLE_API_KEY del entorno)
llm_validador = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
llm_principal = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# Prompt del clasificador de seguridad
prompt_clasificador = ChatPromptTemplate.from_messages([
    ("system", "Clasifica la pregunta. Responde estrictamente 'PERMITIDO' si es de soporte técnico, o 'PROHIBIDO' para cualquier otro tema."),
    ("user", "{input}")
])
cadena_clasificadora = prompt_clasificador | llm_validador | StrOutputParser()

# Lógica del asistente principal
prompt_principal = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente de soporte técnico amigable. Ayuda al usuario."),
    ("user", "{input}")
])
cadena_principal = prompt_principal | llm_principal | StrOutputParser()

# Acción en caso de bloqueo
cadena_rechazo = RunnablePassthrough() | (lambda x: "Lo siento, solo puedo responder dudas de soporte técnico.")

# Enrutamiento dinámico LCEL
cadena_segura = RunnableBranch(
    (lambda x: "PROHIBIDO" in cadena_clasificadora.invoke({"input": x["input"]}), cadena_rechazo),
    cadena_principal
)

# --- Ejecución ---
if __name__ == "__main__":
    print(cadena_segura.invoke({"input": "¿Cómo configuro el correo en mi celular?"}))
