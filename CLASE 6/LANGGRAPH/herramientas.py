from langchain_core.tools import tool

@tool
def buscar_inventario_producto(producto: str) -> str:
    """Busca la disponibilidad y stock de un producto específico en la tienda."""
    inventario = {"laptop": "5 unidades disponibles, precio $800", "mouse": "0 unidades (Agotado)"}
    return inventario.get(producto.lower(), "Producto no encontrado en el sistema.")

@tool
def calcular_descuento_envio(total: float) -> str:
    """Calcula el costo de envío. Si la compra es mayor a $500, el envío es gratis."""
    if total > 500:
        return "Envío GRATIS aplicado."
    return "Costo de envío estándar: $15."

# Lista exportable de herramientas
tools = [buscar_inventario_producto, calcular_descuento_envio]
