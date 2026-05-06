"""
Funciones auxiliares para el procesamiento de imagenes.
Incluye operaciones de clamp, padding y normalizacion.
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


def normalizar_imagen(imagen):
    """
    Normaliza los valores de la imagen al rango [0, 255].
    Util despues de operaciones que producen valores fuera de rango.
    """
    minimo = imagen.min()
    maximo = imagen.max()
    if maximo - minimo == 0:
        return np.zeros_like(imagen, dtype=np.uint8)
    normalizada = (imagen - minimo) / (maximo - minimo) * 255.0
    resultado = np.zeros_like(normalizada, dtype=np.uint8)
    alto, ancho = normalizada.shape
    for i in range(alto):
        for j in range(ancho):
            resultado[i, j] = int(clamp(round(normalizada[i, j]), 0, 255))
    return resultado


def imagen_a_float(imagen):
    """Convierte imagen uint8 a float64 para calculos."""
    return imagen.astype(np.float64)


def float_a_imagen(imagen):
    """Convierte imagen float64 a uint8, clampeando valores."""
    resultado = np.zeros_like(imagen, dtype=np.uint8)
    alto, ancho = imagen.shape
    for i in range(alto):
        for j in range(ancho):
            resultado[i, j] = int(clamp(round(imagen[i, j]), 0, 255))
    return resultado
