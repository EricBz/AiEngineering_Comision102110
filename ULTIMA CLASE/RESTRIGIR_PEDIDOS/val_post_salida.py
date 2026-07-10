from langchain_core.runnables import RunnableLambda

PALABRAS_PROHIBIDAS = ["competidorX", "secreto123", "gratis"]

def verificar_politicas(modelo_output: str) -> str:
    # Si la respuesta del LLM tiene texto prohibido, la censuramos
    if any(palabra in modelo_output.lower() for palabra in PALABRAS_PROHIBIDAS):
        return "Lo siento, la respuesta generada infringe nuestras políticas de seguridad."
    return modelo_output

# Añadimos la función al final de la cadena de ejecución
#cadena_con_filtro_salida = prompt_principal | llm_principal | StrOutputParser() | RunnableLambda(verificar_politicas)
