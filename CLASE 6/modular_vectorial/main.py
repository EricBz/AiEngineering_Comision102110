from graph_builder import graph
from persistence import load_state

print("=== Agente Inteligente de Soporte ===\n")

recover = load_state()

if recover and not recover["finished"]:
    print("Se encontró un checkpoint previo.")
    answer = input("¿Desea continuar la ejecución? (s/n): ")

    if answer.lower() == "s":
        state = recover
    else:
        state = None
else:
    state = None

if state is None:
    question = input("Ingrese una consulta: ")
    #question = "¿Cómo puedo restablecer mi contraseña?"

    state = {
        "question": question,
        "retrieved_docs": [],
        "answer": "",
        "attempts": 0,
        "finished": False
    }

result = graph.invoke(state)

print("\nRespuesta:\n")
print(result["answer"])