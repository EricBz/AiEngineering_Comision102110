from langgraph.graph import StateGraph, MessagesState, START, END

# 1. Definir la función que procesa la información
def mock_llm(state: MessagesState):
    # Retorna un diccionario con la estructura que MessagesState espera para añadir un mensaje
    return {"messages": [{"role": "ai", "content": "hello world"}]}

def bye(state: MessagesState):
    return {"messages": [{"role":"ai", "content": "bye my friend!"}]}

# 2. Construir el grafo de flujo
builder = StateGraph(MessagesState)
builder.add_node("mock_llm", mock_llm) # Añade el nodo explícitamente con un nombre
builder.add_node("bye", bye)
builder.add_edge(START, "mock_llm")
builder.add_edge("mock_llm","bye")    # Flujo de inicio al nodo
builder.add_edge("bye", END)      # Flujo del nodo al final

# 3. Compilar el grafo
graph = builder.compile()

terminal = input()
# 4. Definir los datos de entrada (un mensaje del usuario)
entrada = {"messages": [{"role": "user", "content": f"Ingresa tu mensaje: {terminal}"}]}

# 5. Ejecutar e imprimir el resultado final

resultado = graph.invoke(entrada)
print(resultado)
