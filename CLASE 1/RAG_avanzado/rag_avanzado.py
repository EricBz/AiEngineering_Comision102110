with open("articulo.txt", "r", encoding="utf-8") as archivo:
    contenido = archivo.read()

# Divide el texto en fragmentos (chunks) usando los dobles saltos de línea (\n\n).
# .strip() elimina espacios y saltos de línea sobrantes al inicio y al final de cada fragmento.
# "if c.strip()" actúa como filtro para ignorar y descartar fragmentos que estén vacíos.
chunks = [
    c.strip()
    for c in contenido.split("\n\n")
    if c.strip()
]

import chromadb
from sentence_transformers import SentenceTransformer
#carga un modelo de inteligencia artificial que convierte texto en listas de números (vectores)
modelo = SentenceTransformer(
    "all-MiniLM-L6-v2"
)
# Inicializa el cliente de ChromaDB en memoria para gestionar la base de datos vectorial.
cliente = chromadb.Client()
# Crea una nueva colección en la base de datos llamada "docs" para almacenar los fragmentos.
coleccion = cliente.create_collection(
    name="docs"
)
# Recorre los fragmentos. enumerate() devuelve tanto el índice (i) como el elemento (chunk).
for i, chunk in enumerate(chunks):
 # modelo.encode() genera el embedding (un array de NumPy con la representación matemática).
    # .tolist() convierte ese array de NumPy en una lista estándar de Python para poder guardarla.
    embedding = modelo.encode(
        chunk
    ).tolist()
    # Guarda el fragmento en ChromaDB. 
    # str(i) convierte el índice numérico en un string de texto, ya que los IDs deben ser texto.
    coleccion.add(
        ids=[str(i)],
        documents=[chunk],
        embeddings=[embedding]
    )

query = "¿Qué base de datos sirve para RAG?"
query_embedding = modelo.encode(
    query
).tolist()
# Convierte la pregunta en un vector numérico y .tolist() lo transforma en lista estándar.
# Realiza la búsqueda vectorial en ChromaDB comparando el embedding de la consulta.
# n_results=3 indica que la salida entregará como máximo los 3 fragmentos más parecidos.
resultado_vectorial = coleccion.query(
    query_embeddings=[query_embedding],
    n_results=3
)

#BUSQUEDA POR PALABRA (BM25) ---------------------------------------------------------------------
from rank_bm25 import BM25Okapi
# Prepara los textos para el algoritmo BM25.
# .lower() pasa el texto a minúsculas y .split() lo divide en palabras individuales (tokens).
tokenizados = [
    chunk.lower().split()
    for chunk in chunks
]
# Inicializa el motor BM25 con los fragmentos ya tokenizados.
bm25 = BM25Okapi(tokenizados)
# Calcula la puntuación de relevancia de cada fragmento respecto a las palabras de la consulta.
scores = bm25.get_scores(
    query.lower().split()
)
#Ordena los fragmentos según su puntuación BM25.
# range(len(scores)) genera los índices. sorted() los ordena basándose en el valor de scores[i].
# reverse=True hace que vaya de mayor a menor. [:3] extrae solo los 3 mejores índices de salida.
indices_bm25 = sorted(
    range(len(scores)),
    key=lambda i: scores[i],
    reverse=True
)[:3]

#COMBINAR
# COMBINAR
# Se usa un conjunto (set) para almacenar los candidatos. 
# El set elimina automáticamente los duplicados si un fragmento fue seleccionado por ambos métodos.
candidatos = set()
# Extrae los documentos devueltos por la búsqueda vectorial y los añade al conjunto.
for doc in resultado_vectorial["documents"][0]:
    candidatos.add(doc)
# Usa los índices obtenidos en BM25 para buscar los fragmentos originales y añadirlos al conjunto.
for idx in indices_bm25:
    candidatos.add(chunks[idx])
# Convierte el conjunto (set) de nuevo en una lista para poder iterar y ordenar en los siguientes pasos.
candidatos = list(candidatos)

#print(candidatos)
#Re - ranking ----------------------------------------------------------------------------------

from sklearn.metrics.pairwise import cosine_similarity
# Genera el embedding vectorial de la consulta original.
query_emb = modelo.encode(query)
# Lista vacía donde se guardará el ordenamiento final de los fragmentos.
ranking = []
# Recorre la lista unificada de candidatos para evaluar su similitud final.
for doc in candidatos:
    # Genera el embedding del fragmento actual evaluado.
    doc_emb = modelo.encode(doc)
 # Mide la similitud de coseno entre el vector de la consulta y el del fragmento.
    # [0][0] extrae el valor flotante directo de la matriz bidimensional que devuelve scikit-learn.
    score = cosine_similarity(
        [query_emb],
        [doc_emb]
    )[0][0]
    # Agrega una tupla (fragmento, puntuación) a la lista de ranking.
    ranking.append(
        (doc, score)
    )
# Ordena la lista de salida de forma final (aquí completamos la sintaxis de tu código).
# key=lambda x: x[1] le dice a Python que ordene usando el score (segundo elemento de la tupla).
# reverse=True asegura que los fragmentos más relevantes y con mayor puntaje queden al principio.
ranking.sort(
    key=lambda x: x[1],
    reverse=True
)

mejor_contexto = ranking[0][0]

#print(mejor_contexto)

#Enviamos a Gemini ---------------------------------------------------------------------------------
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(
    api_key=API_KEY
)

prompt = f""" Respondeme en un frase.
Contexto:
{mejor_contexto}

Pregunta:
{query}
"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)

print(response.text)

