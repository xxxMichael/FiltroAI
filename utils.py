"""
Funciones auxiliares para el procesamiento de imagenes.
Incluye operaciones de clamp y padding.
"""

import numpy as np


def clamp(valor, minimo=0, maximo=255):
    """Limita un valor al rango [minimo, maximo]."""
    if valor < minimo:
        return minimo
    if valor > maximo:
        return maximo
    return valor


def zero_pad(imagen, pad):
    """
    Agrega padding de ceros alrededor de la imagen.
    
    Parametros:
        imagen: matriz 2D (numpy array)
        pad: cantidad de pixeles de padding a agregar por lado
    
    Retorna:
        Imagen con padding de ceros
    """
    alto, ancho = imagen.shape
    nueva = np.zeros((alto + 2 * pad, ancho + 2 * pad), dtype=np.float64)
    nueva[pad:pad + alto, pad:pad + ancho] = imagen
    return nueva
