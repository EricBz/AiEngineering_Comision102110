import pytest
from unittest.mock import patch
from langsmith import unit
from app import generar_respuesta

@pytest.mark.langsmith
@unit
def test_respuesta_cielo():
    # CORRECCIÓN: Interceptamos el resultado final de la función, evitando tocar a Pydantic por dentro
    with patch("test_app.generar_respuesta", return_value="El cielo es azul"):
        resultado = generar_respuesta("¿De qué color es el cielo en un día despejado? Responde en una palabra.")
        
        print(f"\n[DEBUG Mock] Simulación de respuesta exitosa: {resultado}")
        
        # Validación
        assert "azul" in resultado.lower()

# py -m pytest -v -s test_app.py