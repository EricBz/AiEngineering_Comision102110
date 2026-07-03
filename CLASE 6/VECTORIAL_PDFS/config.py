"""
config.py

Configuración general del proyecto.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ==========================
# API KEY
# ==========================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ==========================
# MODELOS
# ==========================

CHAT_MODEL = "gemini-2.5-flash"

# Modelo de embeddings vigente de Google
EMBEDDING_MODEL = "gemini-embedding-001"

# ==========================
# CHROMA
# ==========================

CHROMA_PATH = "./chroma_db"

COLLECTION_NAME = "pdf_collection"

# ==========================
# DOCUMENTOS
# ==========================

DOCS_PATH = "./docs"

# ==========================
# RECUPERACIÓN
# ==========================

TOP_K = 20

RERANK_TOP_K = 5