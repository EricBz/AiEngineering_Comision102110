#En este caso se ve que el agente usa las herramientas de manera condicional,
#por eso mismo ahora si necesitara el uso de un llm uqe decida, se debera agregar docstyles,

import os
from dotenv import load_dotenv
from typing import Literal
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# 1. Definir la lógica pura de la función de despedida
def bye_function(name: str) -> str:
    return f"¡Adiós {name}! Que tengas un excelente día."

# Construir la herramienta de manera explícita sin decoradores
bye_tool = StructuredTool.from_function(
    func=bye_function,
    name="bye",
    description="Utiliza esta herramienta obligatoriamente cuando el usuario se esté despidiendo o diga adiós."
)

tools = [bye_tool]

# 2. Inicializar Gemini y enlazar las herramientas de forma nativa
# Recuerda configurar tu entorno: os.environ["GEMINI_API_KEY"] = "tu-api-key"
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("Falta la variable de entorno GOOGLE_API_KEY en tu archivo .env")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# 3. Nodos del Grafo
def assistant(state: MessagesState):
    """Procesa el estado actual y genera una respuesta o llamada a herramienta."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Instanciar el nodo que ejecutará las herramientas declaradas
tool_node = ToolNode(tools)

# 4. Construcción del Grafo utilizando la sintaxis estricta y limpia
builder = StateGraph(MessagesState)

# Registrar los componentes
builder.add_node("assistant", assistant)
builder.add_node("tools", tool_node)

# Flujo inicial
builder.add_edge(START, "assistant")

# Condicional moderna utilizando un mapa de rutas (path_map) explícito
builder.add_conditional_edges(
    "assistant",
    tools_condition,
    path_map={
        "tools": "tools",  # Si tools_condition detecta herramienta, va al nodo tools
        "__end__": END     # Si tools_condition no detecta nada, finaliza el flujo
    }
)

# El nodo de herramientas siempre regresa al asistente para dar la respuesta final
builder.add_edge("tools", "assistant")

# Compilar
graph = builder.compile()

# 5. Ejecución de pruebas
print("--- Caso 1: Saludo ordinario (No activa herramienta) ---")
res1 = graph.invoke({"messages": [{"role": "user", "content": "Hola, ¿cómo estás?"}]})
print(res1["messages"][-1].content)

print("\n--- Caso 2: Despedida (Activa la herramienta 'bye') ---")
res2 = graph.invoke({"messages": [{"role": "user", "content": "Chao, me voy a dormir, me llamo Juan."}]})
print(res2["messages"][-1].content)
