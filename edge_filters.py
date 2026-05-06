"""
Filtros pasa-altos para acentuado de bordes.
Incluye kernels Laplaciano y Sobel implementados con convolucion manual.
"""

import numpy as np
from spatial_filters import convolucion_manual
from utils import normalizar_imagen, clamp


def crear_kernel_laplaciano():
    """
    Crea un kernel Laplaciano 3x3 para deteccion de bordes.
    
    Kernel:
    [ -1, -1, -1 ]
    [ -1,  8, -1 ]
    [ -1, -1, -1 ]
    
    El valor central (8) menos la suma de los vecinos (-8) da un efecto
    de sustraccion que acentua los cambios bruscos de intensidad.
    """
    kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=np.float64)
    return kernel


def crear_kernel_sobel_horizontal():
    """
    Kernel Sobel para deteccion de bordes horizontales.
    
    Kernel:
    [ -1, -2, -1 ]
    [  0,  0,  0 ]
    [  1,  2,  1 ]
    """
    kernel = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float64)
    return kernel


def crear_kernel_sobel_vertical():
    """
    Kernel Sobel para deteccion de bordes verticales.
    
    Kernel:
    [ -1,  0,  1 ]
    [ -2,  0,  2 ]
    [ -1,  0,  1 ]
    """
    kernel = np.array([
        [-1,  0,  1],
        [-2,  0,  2],
        [-1,  0,  1]
    ], dtype=np.float64)
    return kernel


def filtro_pasa_altos(imagen):
    """
    Aplica el filtro pasa-altos Laplaciano para acentuar bordes.
    
    Parametros:
        imagen: numpy array 2D uint8
    
    Retorna:
        Imagen con bordes acentuados (numpy array 2D uint8)
    """
    kernel = crear_kernel_laplaciano()
    bordes = convolucion_manual(imagen.astype(np.float64), kernel)
    return normalizar_imagen(bordes)


def acentuar_bordes(imagen_suavizada, alpha=0.5):
    """
    Acentua los bordes de una imagen suavizada.
    
    Proceso:
    1. Detectar bordes con kernel Laplaciano
    2. Sumar bordes a la imagen suavizada: resultado = suavizada + alpha * bordes
    
    Parametros:
        imagen_suavizada: imagen ya filtrada (limpia de ruido)
        alpha: factor de intensidad del acentuado (0.0 a 2.0)
    
    Retorna:
        Imagen con bordes acentuados (numpy array 2D uint8)
    """
    kernel = crear_kernel_laplaciano()
    bordes = convolucion_manual(imagen_suavizada.astype(np.float64), kernel)
    
    alto, ancho = imagen_suavizada.shape
    resultado = np.zeros((alto, ancho), dtype=np.uint8)
    
    for i in range(alto):
        for j in range(ancho):
            valor = float(imagen_suavizada[i, j]) + alpha * bordes[i, j]
            resultado[i, j] = int(clamp(round(valor), 0, 255))
    
    return resultado


def deteccion_bordes_sobel(imagen):
    """
    Detecta bordes usando gradientes Sobel (horizontal + vertical).
    
    Magnitud del gradiente: G = sqrt(Gx^2 + Gy^2)
    
    Parametros:
        imagen: numpy array 2D uint8
    
    Retorna:
        Mapa de bordes (numpy array 2D uint8)
    """
    import math
    
    kernel_h = crear_kernel_sobel_horizontal()
    kernel_v = crear_kernel_sobel_vertical()
    
    gx = convolucion_manual(imagen.astype(np.float64), kernel_h)
    gy = convolucion_manual(imagen.astype(np.float64), kernel_v)
    
    alto, ancho = imagen.shape
    magnitud = np.zeros((alto, ancho), dtype=np.float64)
    
    for i in range(alto):
        for j in range(ancho):
            magnitud[i, j] = math.sqrt(gx[i, j] ** 2 + gy[i, j] ** 2)
    
    return normalizar_imagen(magnitud)
