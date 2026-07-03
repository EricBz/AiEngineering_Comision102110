from dotenv import load_dotenv
load_dotenv()

from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_google_genai import ChatGoogleGenerativeAI
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
from app import generar_respuesta

client = Client()
dataset_name = "Dataset de Prueba Gemini"

if not client.has_dataset(dataset_name=dataset_name):
    dataset = client.create_dataset(dataset_name=dataset_name)
    client.create_example(
        inputs={"question": "¿De qué color es el cielo?"}, 
        outputs={"answer": "Azul"}, 
        dataset_id=dataset.id
    )

def evaluar_app(inputs: dict):
    resultado = generar_respuesta(inputs["question"])
    return {"output": resultado}

if __name__ == "__main__":
    juez_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    evaluador_correccion = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        judge_model=juez_llm
    )
    
    print("Iniciando evaluación en LangSmith con Gemini 2.5 Flash...")
    evaluate(
        evaluar_app,
        data=dataset_name,
        evaluators=[evaluador_correccion],
        experiment_prefix="eval-gemini-2.5-flash"
    )
    print("¡Evaluación completada con éxito!")
