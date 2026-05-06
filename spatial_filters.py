"""
Filtros en el dominio espacial implementados manualmente.
Incluye filtros de media, mediana y moda con convolucion manual.
NO se usan funciones predefinidas de convolucion ni estadisticas.
"""

import numpy as np
from utils import zero_pad, clamp


def filtro_media(imagen, tamano_kernel=3):
    """
    Filtro de media (promedio) implementado con convolucion manual.
    
    Para cada pixel (x,y), calcula el promedio de los pixeles
    en la ventana de tamano kernel_size x kernel_size centrada en (x,y).
    
    Formula: g(x,y) = (1/N) * sum de f(i,j) en la vecindad
    
    Parametros:
        imagen: numpy array 2D con valores uint8
        tamano_kernel: tamano de la ventana (debe ser impar)
    
    Retorna:
        Imagen filtrada (numpy array 2D uint8)
    """
    alto, ancho = imagen.shape
    pad = tamano_kernel // 2
    imagen_pad = zero_pad(imagen.astype(np.float64), pad)
    resultado = np.zeros((alto, ancho), dtype=np.uint8)
    
    total_celdas = tamano_kernel * tamano_kernel
    
    for i in range(alto):
        for j in range(ancho):
            suma = 0.0
            for ki in range(tamano_kernel):
                for kj in range(tamano_kernel):
                    suma += imagen_pad[i + ki, j + kj]
            
            promedio = suma / total_celdas
            resultado[i, j] = int(clamp(round(promedio), 0, 255))
    
    return resultado


def filtro_mediana(imagen, tamano_kernel=3):
    """
    Filtro de mediana implementado manualmente.
    
    Para cada pixel (x,y), recopila los valores de la vecindad,
    los ordena y selecciona el valor central (mediana).
    
    Nota: El filtro de mediana es particularmente efectivo para
    eliminar ruido sal y pimienta.
    
    Parametros:
        imagen: numpy array 2D con valores uint8
        tamano_kernel: tamano de la ventana (debe ser impar)
    
    Retorna:
        Imagen filtrada (numpy array 2D uint8)
    """
    alto, ancho = imagen.shape
    pad = tamano_kernel // 2
    imagen_pad = zero_pad(imagen.astype(np.float64), pad)
    resultado = np.zeros((alto, ancho), dtype=np.uint8)
    
    for i in range(alto):
        for j in range(ancho):
            # Recopilar valores de la vecindad
            vecinos = []
            for ki in range(tamano_kernel):
                for kj in range(tamano_kernel):
                    vecinos.append(imagen_pad[i + ki, j + kj])
            
            # Ordenar manualmente (algoritmo de ordenamiento burbuja)
            n = len(vecinos)
            for a in range(n):
                for b in range(0, n - a - 1):
                    if vecinos[b] > vecinos[b + 1]:
                        vecinos[b], vecinos[b + 1] = vecinos[b + 1], vecinos[b]
            
            # Seleccionar la mediana (valor central)
            mediana = vecinos[n // 2]
            resultado[i, j] = int(clamp(round(mediana), 0, 255))
    
    return resultado


def filtro_moda(imagen, tamano_kernel=3):
    """
    Filtro de moda implementado manualmente.
    
    Para cada pixel (x,y), recopila los valores de la vecindad,
    cuenta la frecuencia de cada valor y selecciona el mas repetido.
    
    En caso de empate, se selecciona el valor mas pequeno entre los
    que tienen la mayor frecuencia.
    
    Parametros:
        imagen: numpy array 2D con valores uint8
        tamano_kernel: tamano de la ventana (debe ser impar)
    
    Retorna:
        Imagen filtrada (numpy array 2D uint8)
    """
    alto, ancho = imagen.shape
    pad = tamano_kernel // 2
    imagen_pad = zero_pad(imagen.astype(np.float64), pad)
    resultado = np.zeros((alto, ancho), dtype=np.uint8)
    
    for i in range(alto):
        for j in range(ancho):
            # Recopilar valores de la vecindad
            vecinos = []
            for ki in range(tamano_kernel):
                for kj in range(tamano_kernel):
                    vecinos.append(int(round(imagen_pad[i + ki, j + kj])))
            
            # Contar frecuencia de cada valor usando diccionario
            frecuencias = {}
            for val in vecinos:
                if val in frecuencias:
                    frecuencias[val] += 1
                else:
                    frecuencias[val] = 1
            
            # Encontrar el valor con mayor frecuencia
            moda_valor = vecinos[0]
            max_frecuencia = 0
            for val, freq in frecuencias.items():
                if freq > max_frecuencia:
                    max_frecuencia = freq
                    moda_valor = val
                elif freq == max_frecuencia and val < moda_valor:
                    # En caso de empate, seleccionar el valor menor
                    moda_valor = val
            
            resultado[i, j] = int(clamp(moda_valor, 0, 255))
    
    return resultado
