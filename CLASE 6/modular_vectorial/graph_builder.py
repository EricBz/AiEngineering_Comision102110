from langgraph.graph import StateGraph, END
from state import AgentState
from rag import search_documents
from persistence import save_state
from audit import log_transition

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

# Nodo 1
def retrieve(state: AgentState):
    log_transition("BUSCAR_DOCUMENTOS")

    docs = search_documents(state["question"])
    state["retrieved_docs"] = docs

    save_state(state)
    return state

# Nodo 2
def generate(state: AgentState):
    log_transition("GENERAR_RESPUESTA")

    context = "\n".join(state["retrieved_docs"])

    prompt = f"""
    Responde utilizando únicamente el siguiente contexto.

    Contexto:
    {context}

    Pregunta:
    {state["question"]}
    """

    response = llm.invoke(prompt)

    state["answer"] = response.content
    save_state(state)

    return state

# Nodo 3
def evaluate(state: AgentState):
    log_transition("EVALUAR")

    if state["retrieved_docs"]:
        state["finished"] = True
        return "finish"

    state["attempts"] += 1

    if state["attempts"] >= 3:
        return "human"

    return "retry"

# Nodo 4
def refine(state: AgentState):
    log_transition("REFINAR")

    state["question"] += " explicación"
    save_state(state)

    return state

# Nodo 5
def human(state: AgentState):
    log_transition("ESCALAR_A_HUMANO")

    state["answer"] = (
        "No se encontró una respuesta adecuada. "
        "El caso debe ser revisado por un operador humano."
    )

    state["finished"] = True
    save_state(state)

    return state


builder = StateGraph(AgentState)

builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_node("refine", refine)
builder.add_node("human", human)

builder.set_entry_point("retrieve")

builder.add_edge("retrieve", "generate")

builder.add_conditional_edges(
    "generate",
    evaluate,
    {
        "finish": END,
        "retry": "refine",
        "human": "human"
    }
)

builder.add_edge("refine", "retrieve")
builder.add_edge("human", END)

graph = builder.compile()