from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    calificacion: str  # Guardará "Aprobado" o "Rechazado"
    feedback: str      # Notas para mejorar la respuesta si fue rechazada
