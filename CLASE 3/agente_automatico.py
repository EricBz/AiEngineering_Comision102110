from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
# IMPORTANTE: Cambia el import deprecado por el nuevo estándar de LangChain
from langchain.agents import create_agent 

# 1. Definir las herramientas igual que antes
@tool
def greet_user(username: str) -> str:
    """Utiliza esta herramienta obligatoriamente cuando el usuario te salude por primera vez o te diga hola."""
    return f"¡Hola {username}! Bienvenido de nuevo. ¿En qué te puedo colaborar hoy?"

@tool
def bye_user(username: str) -> str:
    """Utiliza esta herramienta obligatoriamente cuando el usuario se esté despidiendo, diga adiós o chao."""
    return f"¡Adiós {username}! Que tengas un excelente día."

tools = [greet_user, bye_user]

# 2. Inicializar Gemini
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 3. EL CAMBIO: Usar la nueva factoría oficial de agentes
# create_agent reemplaza de forma idéntica a create_react_agent, pero por dentro
# utiliza un sistema de middleware mucho más rápido y estable.
graph = create_agent(llm, tools)

# 4. Probar la ejecución
entrada = {"messages": [{"role": "user", "content": "Hola, me llamo Mateo"}]}
res = graph.invoke(entrada)

print(res["messages"][-1].content)


# ==========================================
# PRUEBAS DE ENRUTAMIENTO AUTOMÁTICO
# ==========================================

print("--- Caso 1: Activación del Saludo ---")
entrada_saludo = {"messages": [{"role": "user", "content": "Hola buenas, me llamo Mateo"}]}
res1 = graph.invoke(entrada_saludo)
print(res1["messages"][-1].content)


print("\n--- Caso 2: Conversación Ordinaria (Sin herramientas) ---")
entrada_charla = {"messages": [{"role": "user", "content": "¿De qué color es el cielo de día?"}]}
res2 = graph.invoke(entrada_charla)
print(res2["messages"][-1].content)


print("\n--- Caso 3: Activación de la Despedida ---")
entrada_despedida = {"messages": [{"role": "user", "content": "Chao, ya me tengo que retirar. Soy Mateo."}]}
res3 = graph.invoke(entrada_despedida)
print(res3["messages"][-1].content)
