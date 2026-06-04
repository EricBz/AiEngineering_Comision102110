import os
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv

load_dotenv()

# 1. Configurar la API Key de Gemini en el entorno
API_KEY = os.getenv("GEMINI_API_KEY")
os.environ["GEMINI_API_KEY"] = API_KEY

# 2. Inicializar el cliente de ChromaDB local
client = chromadb.PersistentClient(path="./mi_base_vectorial_gemini")

# 3. Definir la función usando el modelo activo 'gemini-embedding-001'
funcion_gemini = embedding_functions.GoogleGeminiEmbeddingFunction(
    model_name="gemini-embedding-001",
    task_type="RETRIEVAL_DOCUMENT"
)

# 4. Crear o cargar la colección
coleccion = client.get_or_create_collection(
    name="documentos_gemini",
    embedding_function=funcion_gemini
)

# 5. Agregar documentos
# Nota: Si el ID ya existe localmente, ChromaDB lo actualizará o ignorará
coleccion.add(
    documents=[
        "El aprendizaje profundo (deep learning) y las redes neuronales son subcampos de la inteligencia artificial.",
        "Las manzanas, plátanos y naranjas son frutas deliciosas.",
        "Los algoritmos de búsqueda vectorial encuentran datos por su significado semántico."
    ],
    ids=["id1", "id2", "id3"]
)

# 6. Consultar
resultado = coleccion.query(
    query_texts=["¿Qué es el aprendizaje profundo?"],
    n_results=1
)

print("Documento más cercano encontrado:")
print(resultado["documents"])
