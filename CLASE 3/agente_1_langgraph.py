import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver  # Persistencia nativa verificada en 2026

# 1. Cargar configuración desde el archivo .env
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("Falta la variable de entorno GOOGLE_API_KEY en tu archivo .env")

# 2. Definir el Estado Compartido (El Array Centralizado)
class State(TypedDict):
    # 'add_messages' asegura que cada turno se agregue al array sin borrar el pasado
    messages: Annotated[list, add_messages]

# 3. Inicializar el LLM de Google
# Usamos gemini-1.5-flash optimizado para respuestas rápidas por streaming
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# 4. Definir el Nodo del Chatbot
def chatbot(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# 5. Construcción y compilación del Grafo
workflow = StateGraph(State)
workflow.add_node("chatbot", chatbot)
workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot", END)

# El checkpointer de memoria actúa como el almacén interno del array por hilos
memory_checkpointer = MemorySaver()
agent = workflow.compile(checkpointer=memory_checkpointer)

# --- INTERFAZ INTERACTIVA POR ENTRADAS (INPUTS) ---
def iniciar_chat():
    # Asignamos un thread_id fijo para mantener la persistencia en esta sesión
    config = {"configurable": {"thread_id": "sesion_interactiva_terminal"}}
    
    print("=" * 60)
    print("🤖 ¡Agente LangGraph + Gemini Listo! (Estándar 2026)")
    print("Escribe tu mensaje. Para salir, escribe 'salir', 'quit' o 'exit'.")
    print("=" * 60)
    
    while True:
        try:
            # Capturar la entrada del usuario
            usuario_input = input("\n👤 Tú: ").strip()
            
            # Condición de salida
            if usuario_input.lower() in ["salir", "quit", "exit"]:
                print("🤖 Agente: ¡Hasta luego!")
                break
                
            if not usuario_input:
                continue

            # Crear el mensaje humano estructurado
            nuevo_mensaje = HumanMessage(content=usuario_input)
            
            print("🤖 Agente: ", end="", flush=True)
            
            # Ejecutar el flujo del grafo en modo streaming a través del estado
            # Pasamos el nuevo mensaje y la configuración que contiene nuestro hilo de memoria
            for event in agent.stream({"messages": [nuevo_mensaje]}, config):
                for value in event.values():
                    # Obtenemos el último mensaje generado en el estado por el nodo 'chatbot'
                    ultimo_msg = value["messages"][-1]
                    if isinstance(ultimo_msg, AIMessage):
                        print(ultimo_msg.content, end="", flush=True)
            print() # Salto de línea al terminar la respuesta

        except KeyboardInterrupt:
            print("\n🤖 Agente: Conversación finalizada abruptamente.")
            break

if __name__ == "__main__":
    iniciar_chat()
