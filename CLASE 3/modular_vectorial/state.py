from typing import TypedDict, List

class AgentState(TypedDict):
    question: str
    retrieved_docs: List[str]
    answer: str
    attempts: int
    finished: bool