import os
from dotenv import load_dotenv  # <-- NUEVO: Importamos dotenv

# <-- NUEVO: Cargamos las variables de entorno del archivo .env antes de importar LangChain
load_dotenv()

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage

# Importaciones locales
from estado import AgentState
from herramientas import tools, buscar_inventario_producto, calcular_descuento_envio

# 1. Configuración del modelo Gemini (ya detecta automáticamente la variable cargada por dotenv)
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash").bind_tools(tools)

# 2. Definición de los Nodos del Grafo
def llamar_agente(state: AgentState):
    messages = state["messages"]
    if state.get("feedback"):
        messages = messages + [SystemMessage(content=f"ALERTA: Tu respuesta anterior fue rechazada. Corrección necesaria: {state['feedback']}")]
    respuesta = model.invoke(messages)
    return {"messages": [respuesta]}

def ejecutar_herramientas(state: AgentState):
    ultimo_mensaje = state["messages"][-1]
    tool_outputs = []
    for tool_call in ultimo_mensaje.tool_calls:
        herramienta_nom = tool_call["name"]
        argumentos = tool_call["args"]
        
        if herramienta_nom == "buscar_inventario_producto":
            resultado = buscar_inventario_producto.invoke(argumentos)
        elif herramienta_nom == "calcular_descuento_envio":
            resultado = calcular_descuento_envio.invoke(argumentos)
        else:
            resultado = "Herramienta no encontrada."
        tool_outputs.append(ToolMessage(content=str(resultado), tool_call_id=tool_call["id"]))
    return {"messages": tool_outputs}

def calificar_respuesta(state: AgentState):
    print("\n--- [🤖 AGENTE TERMINÓ DE REDACTAR] ---")
    print(f"Respuesta actual de Gemini: {state['messages'][-1].content}\n")
    return {}

# 3. Construcción del Flujo (Grafo)
workflow = StateGraph(AgentState)
workflow.add_node("agente", llamar_agente)
workflow.add_node("herramientas", ejecutar_herramientas)
workflow.add_node("calificador", calificar_respuesta)

workflow.set_entry_point("agente")

# Rutas condicionales
def ruta_despues_de_agente(state: AgentState):
    ultimo_mensaje = state["messages"][-1]
    if ultimo_mensaje.tool_calls:
        return "herramientas"
    return "calificador"

workflow.add_conditional_edges(
    "agente",
    ruta_despues_de_agente,
    {"herramientas": "herramientas", "calificador": "calificador"}
)

workflow.add_edge("herramientas", "agente")

def ruta_despues_de_calificar(state: AgentState):
    if state.get("calificacion") == "Aprobado":
        return "finalizar"
    return "reintentar"

workflow.add_conditional_edges(
    "calificador",
    ruta_despues_de_calificar,
    {"finalizar": END, "reintentar": "agente"}
)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory, interrupt_after=["calificador"])


# 4. Orquestación Ejecución + Intervención Humana
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "sesion_1"}}
    inputs = {"messages": [HumanMessage(content="Quiero comprar una laptop. ¿Tienen stock y cuánto me saldría con el envío?")]}
    
    print("Iniciando el agente...")
    for evento in app.stream(inputs, config):
        pass 
        
    while True:
        decision_humana = input("¿Aprobás esta respuesta? (si / no): ").strip().lower()
        
        if decision_humana == "si":
            app.update_state(config, {"calificacion": "Aprobado", "feedback": ""}, as_node="calificador")
            print("\nRespuesta aprobada por el humano. Finalizando proceso...")
            break
        elif decision_humana == "no":
            comentarios = input("Escribí el feedback para que Gemini mejore: ")
            app.update_state(
                config, 
                {"calificacion": "Rechazado", "feedback": comentarios}, 
                as_node="calificador"
            )
            print("\nFeedback guardado. Reanudando agente para corrección...\n")
            break
        else:
            print("Por favor, ingresá 'si' o 'no'.")

    for evento in app.stream(None, config):
        if "agente" in evento:
            print("\n--- [✨ RESPUESTA FINAL CORREGIDA POR GEMINI] ---")
            print(evento["agente"]["messages"][-1].content)
