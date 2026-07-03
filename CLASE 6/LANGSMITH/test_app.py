import pytest
from langsmith import unit
from app import generar_respuesta # Importamos tu app

@pytest.mark.langsmith
@unit
def test_respuesta_cielo():
    # Ejecutamos la función de nuestra app
    resultado = generar_respuesta("¿De qué color es el cielo en un día despejado? Responde en una palabra.")
    
    # Validación tradicional de código
    assert "azul" in resultado.lower()
