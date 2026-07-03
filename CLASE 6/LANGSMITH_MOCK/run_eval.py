from dotenv import load_dotenv
load_dotenv()

from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()
dataset_name = "Dataset de Prueba Gemini"

# 1. Creamos tu dataset en LangSmith si no existe
if not client.has_dataset(dataset_name=dataset_name):
    dataset = client.create_dataset(dataset_name=dataset_name)
    client.create_example(
        inputs={"question": "¿De qué color es el cielo?"}, 
        outputs={"answer": "Azul"}, 
        dataset_id=dataset.id
    )

# 2. EVALUADOR DE CÓDIGO LOCAL (Cero consumo de API)
def evaluador_exact_match(run, example):
    output_text = run.outputs.get("output", "").lower()
    reference_text = example.outputs.get("answer", "").lower()
    score = 1.0 if reference_text in output_text else 0.0
    return {"key": "exact_match", "score": score}

# 3. MOCK DIRECTO (Simulación pura sin usar librerías de parcheo)
# En lugar de importar 'generar_respuesta', escribimos una simulación directa
# que devuelve el formato exacto que espera LangSmith.
def evaluar_app_mocked(inputs: dict):
    # Simulamos lo que respondería tu función de Gemini
    return {"output": "El cielo es azul"}

# 4. Bloque de ejecución principal
if __name__ == "__main__":
    print("Iniciando evaluación 100% Simulada Local en LangSmith...")
    
    evaluate(
        evaluar_app_mocked,       # Enviamos la función simulada pura
        data=dataset_name,
        evaluators=[evaluador_exact_match],
        experiment_prefix="eval-gemini-sin-cuota"
    )
    
    print("¡Evaluación completada con éxito! Revisa tu panel web de LangSmith.")
