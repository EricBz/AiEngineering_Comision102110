import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware.pii import PIIMiddleware
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Cargar variables de entorno desde el archivo .env
load_dotenv()

# 2. Definimos una herramienta sencilla para el agente
@tool
def procesar_pago_simulado(monto: float) -> str:
    """Simula el procesamiento de un pago. Úsala cuando pidan cobrar o pagar."""
    return f"Pago exitoso por un monto de ${monto}."

tools = [procesar_pago_simulado]

# 3. Inicializamos el modelo Gemini actualizado
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0
)

# 4. Configuramos la pila de PIIMiddleware (Guardrails de privacidad)
pii_guardrails = [
    PIIMiddleware(
        pii_type="credit_card",
        strategy="mask",          # Enmascara tarjetas antes de ir al LLM
        apply_to_input=True,      
        apply_to_output=False
    ),
    PIIMiddleware(
        pii_type="email",
        strategy="hash",          # Ofusca correos convirtiéndolos en un hash determinista
        apply_to_input=True,      
        apply_to_output=True      
    )
]

# 5. Creamos el agente inyectando el middleware.
# NOTA: En la API actual, el parámetro para pasar el LLM es 'model'
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="Eres un asistente administrativo seguro. Procesa las solicitudes de forma eficiente.",
    middleware=pii_guardrails,
    debug=True  # Reemplaza a verbose=True para ver el paso a paso en consola
)

# 6. Bucle interactivo por terminal utilizando la API unificada (.invoke)
if __name__ == "__main__":
    print("=" * 60)
    print("Agente de IA Iniciado (Dotenv cargado & Guardrails PII Activos)")
    print("Escribe 'salir' o 'exit' para terminar el programa.")
    print("=" * 60)
    
    while True:
        try:
            # Captura la entrada del usuario directamente desde la terminal
            user_query = input("\nTú: ")
            
            # Condición de salida limpia del bucle
            if user_query.strip().lower() in ["salir", "exit"]:
                print("Cerrando sesión segura. ¡Hasta luego!")
                break
                
            # Si el usuario presiona enter vacío, saltamos la iteración
            if not user_query.strip():
                continue
                
            print("\n--- [Procesando con Grafo Compilado] ---")
            
            # Cambiado: Llamamos directamente a agent.invoke() pasando la estructura de mensajes estándar
            response = agent.invoke({
                "messages": [{"role": "user", "content": user_query}]
            })
            
            print("\nAsistente:")
            # Extraemos el contenido del último mensaje devuelto por el grafo
            print(response["messages"][-1].content)
            print("-" * 40)
            
        except Exception as e:
            print(f"\n[ERROR] Ocurrió un problema durante el procesamiento: {e}")
