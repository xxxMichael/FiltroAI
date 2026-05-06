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


def normalizar_histograma(imagen):
    """
    Ecualiza el histograma de una imagen en escala de grises manualmente.
    
    Proceso:
    1. Calcular histograma (frecuencia de cada intensidad 0-255)
    2. Calcular histograma acumulado (CDF)
    3. Normalizar el CDF para mapear los valores al rango [0, 255]
    
    Esto redistribuye las intensidades para mejorar el contraste.
    
    Parametros:
        imagen: numpy array 2D uint8 en escala de grises
    
    Retorna:
        Imagen con histograma ecualizado (numpy array 2D uint8)
    """
    alto, ancho = imagen.shape
    total_pixeles = alto * ancho
    
    # 1. Calcular histograma manualmente
    histograma = [0] * 256
    for i in range(alto):
        for j in range(ancho):
            histograma[int(imagen[i, j])] += 1
    
    # 2. Calcular CDF (histograma acumulado)
    cdf = [0] * 256
    cdf[0] = histograma[0]
    for k in range(1, 256):
        cdf[k] = cdf[k - 1] + histograma[k]
    
    # 3. Encontrar el valor minimo del CDF (primer bin no vacio)
    cdf_min = 0
    for k in range(256):
        if cdf[k] > 0:
            cdf_min = cdf[k]
            break
    
    # 4. Crear tabla de mapeo
    mapeo = [0] * 256
    denominador = total_pixeles - cdf_min
    if denominador > 0:
        for k in range(256):
            valor = round((cdf[k] - cdf_min) / denominador * 255.0)
            mapeo[k] = int(clamp(valor, 0, 255))
    
    # 5. Aplicar mapeo a la imagen
    resultado = np.zeros((alto, ancho), dtype=np.uint8)
    for i in range(alto):
        for j in range(ancho):
            resultado[i, j] = mapeo[int(imagen[i, j])]
    
    return resultado


def binarizar(imagen, umbral=None):
    """
    Binariza una imagen en escala de grises.
    
    Si no se proporciona umbral, se calcula automaticamente usando
    el metodo de Otsu (maximiza la varianza entre clases).
    
    Parametros:
        imagen: numpy array 2D uint8
        umbral: valor de corte (0-255). Si es None, se calcula con Otsu.
    
    Retorna:
        Imagen binarizada (0 o 255) como numpy array 2D uint8
    """
    alto, ancho = imagen.shape
    
    # Calcular umbral con metodo de Otsu si no se proporciona
    if umbral is None:
        # Calcular histograma
        histograma = [0] * 256
        total_pixeles = alto * ancho
        for i in range(alto):
            for j in range(ancho):
                histograma[int(imagen[i, j])] += 1
        
        # Metodo de Otsu: encontrar el umbral que maximiza la varianza entre clases
        mejor_umbral = 0
        mejor_varianza = 0.0
        
        for t in range(256):
            # Clase 0 (pixeles <= t) y Clase 1 (pixeles > t)
            w0 = 0  # peso clase 0
            suma0 = 0.0
            w1 = 0  # peso clase 1
            suma1 = 0.0
            
            for k in range(t + 1):
                w0 += histograma[k]
                suma0 += k * histograma[k]
            for k in range(t + 1, 256):
                w1 += histograma[k]
                suma1 += k * histograma[k]
            
            if w0 == 0 or w1 == 0:
                continue
            
            media0 = suma0 / w0
            media1 = suma1 / w1
            
            # Varianza entre clases
            varianza = w0 * w1 * (media0 - media1) ** 2
            
            if varianza > mejor_varianza:
                mejor_varianza = varianza
                mejor_umbral = t
        
        umbral = mejor_umbral
    
    # Aplicar binarizacion
    resultado = np.zeros((alto, ancho), dtype=np.uint8)
    for i in range(alto):
        for j in range(ancho):
            if imagen[i, j] > umbral:
                resultado[i, j] = 255
            else:
                resultado[i, j] = 0
    
    return resultado


def gris_a_rgb(imagen_gris):
    """
    Convierte una imagen en escala de grises a RGB replicando el canal.
    Para la presentacion final en RGB como indica el profesor.
    
    Parametros:
        imagen_gris: numpy array 2D uint8
    
    Retorna:
        numpy array 3D (alto, ancho, 3) uint8
    """
    alto, ancho = imagen_gris.shape
    rgb = np.zeros((alto, ancho, 3), dtype=np.uint8)
    for i in range(alto):
        for j in range(ancho):
            rgb[i, j, 0] = imagen_gris[i, j]
            rgb[i, j, 1] = imagen_gris[i, j]
            rgb[i, j, 2] = imagen_gris[i, j]
    return rgb


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
