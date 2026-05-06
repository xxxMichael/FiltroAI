"""
Modulo de metricas de calidad de imagen.
Calcula MSE y PSNR para comparar imagenes filtradas contra la original.
"""

import numpy as np
import math


def calcular_mse(imagen_original, imagen_filtrada):
    """
    Calcula el Error Cuadratico Medio (MSE) entre dos imagenes.
    
    Formula: MSE = (1/MN) * sum_i sum_j (f(i,j) - g(i,j))^2
    
    Parametros:
        imagen_original: numpy array 2D (referencia)
        imagen_filtrada: numpy array 2D (resultado del filtro)
    
    Retorna:
        Valor MSE (float)
    """
    alto, ancho = imagen_original.shape
    suma_errores = 0.0
    
    for i in range(alto):
        for j in range(ancho):
            diferencia = float(imagen_original[i, j]) - float(imagen_filtrada[i, j])
            suma_errores += diferencia * diferencia
    
    mse = suma_errores / (alto * ancho)
    return mse


def calcular_psnr(imagen_original, imagen_filtrada):
    """
    Calcula la Relacion Senal/Ruido Pico (PSNR) entre dos imagenes.
    
    Formula: PSNR = 10 * log10(255^2 / MSE)
    
    Parametros:
        imagen_original: numpy array 2D (referencia)
        imagen_filtrada: numpy array 2D (resultado del filtro)
    
    Retorna:
        Valor PSNR en dB (float). Retorna infinito si MSE = 0.
    """
    mse = calcular_mse(imagen_original, imagen_filtrada)
    
    if mse == 0:
        return float('inf')  # Imagenes identicas
    
    psnr = 10.0 * math.log10((255.0 ** 2) / mse)
    return psnr
