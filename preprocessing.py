"""
Modulo de preprocesamiento de imagenes.
Conversion a escala de grises e inyeccion de ruido sal y pimienta.
"""

import numpy as np
from utils import clamp


def convertir_a_grises(imagen_rgb):
    """
    Convierte una imagen RGB a escala de grises usando la formula de luminancia.
    
    Formula: gris = 0.299*R + 0.587*G + 0.114*B
    
    Parametros:
        imagen_rgb: numpy array de forma (alto, ancho, 3) con valores uint8
    
    Retorna:
        numpy array 2D (alto, ancho) con valores uint8
    """
    alto, ancho = imagen_rgb.shape[0], imagen_rgb.shape[1]
    gris = np.zeros((alto, ancho), dtype=np.uint8)
    
    for i in range(alto):
        for j in range(ancho):
            r = float(imagen_rgb[i, j, 0])
            g = float(imagen_rgb[i, j, 1])
            b = float(imagen_rgb[i, j, 2])
            valor = 0.299 * r + 0.587 * g + 0.114 * b
            gris[i, j] = int(clamp(round(valor), 0, 255))
    
    return gris


def agregar_ruido_sal_pimienta(imagen, porcentaje):
    """
    Agrega ruido sal y pimienta a una imagen en escala de grises.
    
    El ruido consiste en pixeles aleatorios que se convierten en:
    - Sal (blanco): valor = 255
    - Pimienta (negro): valor = 0
    
    Parametros:
        imagen: numpy array 2D con valores uint8
        porcentaje: porcentaje de pixeles a afectar (0-100)
    
    Retorna:
        Imagen con ruido agregado (copia, no modifica la original)
    """
    ruidosa = imagen.copy()
    alto, ancho = imagen.shape
    total_pixeles = alto * ancho
    num_pixeles_ruido = int(total_pixeles * porcentaje / 100.0)
    
    # Generar posiciones aleatorias
    np.random.seed(None)  # Semilla aleatoria para cada ejecucion
    
    for _ in range(num_pixeles_ruido):
        fila = np.random.randint(0, alto)
        col = np.random.randint(0, ancho)
        
        # 50% probabilidad de sal (255) o pimienta (0)
        if np.random.random() < 0.5:
            ruidosa[fila, col] = 0    # Pimienta
        else:
            ruidosa[fila, col] = 255  # Sal
    
    return ruidosa
