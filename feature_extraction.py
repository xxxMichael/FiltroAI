"""
Modulo de extraccion de caracteristicas (descriptores).
Calcula vectores de caracteristicas para imagenes filtradas.
Todos los calculos se realizan manualmente sin funciones estadisticas predefinidas.
"""

import numpy as np
import math


def calcular_histograma(imagen):
    """
    Calcula el histograma de intensidades de la imagen manualmente.
    
    Parametros:
        imagen: numpy array 2D uint8
    
    Retorna:
        histograma: lista de 256 elementos con la frecuencia de cada intensidad
    """
    histograma = [0] * 256
    alto, ancho = imagen.shape
    
    for i in range(alto):
        for j in range(ancho):
            valor = int(imagen[i, j])
            histograma[valor] += 1
    
    return histograma


def calcular_media(imagen):
    """
    Calcula la media de intensidad de la imagen manualmente.
    
    Formula: media = (1/N) * sum(pixeles)
    """
    alto, ancho = imagen.shape
    total = alto * ancho
    suma = 0.0
    
    for i in range(alto):
        for j in range(ancho):
            suma += float(imagen[i, j])
    
    return suma / total


def calcular_desviacion_estandar(imagen):
    """
    Calcula la desviacion estandar de intensidad manualmente.
    
    Formula: std = sqrt((1/N) * sum((pixel - media)^2))
    """
    media = calcular_media(imagen)
    alto, ancho = imagen.shape
    total = alto * ancho
    suma_cuadrados = 0.0
    
    for i in range(alto):
        for j in range(ancho):
            diferencia = float(imagen[i, j]) - media
            suma_cuadrados += diferencia * diferencia
    
    varianza = suma_cuadrados / total
    return math.sqrt(varianza)


def calcular_skewness(imagen):
    """
    Calcula el coeficiente de asimetria (skewness) manualmente.
    
    Formula: skewness = (1/N) * sum(((pixel - media) / std)^3)
    
    Mide la asimetria de la distribucion de intensidades.
    Valor positivo = cola derecha mas larga.
    Valor negativo = cola izquierda mas larga.
    """
    media = calcular_media(imagen)
    std = calcular_desviacion_estandar(imagen)
    
    if std == 0:
        return 0.0
    
    alto, ancho = imagen.shape
    total = alto * ancho
    suma = 0.0
    
    for i in range(alto):
        for j in range(ancho):
            valor_normalizado = (float(imagen[i, j]) - media) / std
            suma += valor_normalizado ** 3
    
    return suma / total


def calcular_kurtosis(imagen):
    """
    Calcula la curtosis manualmente.
    
    Formula: kurtosis = (1/N) * sum(((pixel - media) / std)^4) - 3
    
    Mide el grado de apuntamiento de la distribucion.
    Kurtosis > 0: distribucion leptocurtica (mas puntiaguda).
    Kurtosis < 0: distribucion platicurtica (mas plana).
    """
    media = calcular_media(imagen)
    std = calcular_desviacion_estandar(imagen)
    
    if std == 0:
        return 0.0
    
    alto, ancho = imagen.shape
    total = alto * ancho
    suma = 0.0
    
    for i in range(alto):
        for j in range(ancho):
            valor_normalizado = (float(imagen[i, j]) - media) / std
            suma += valor_normalizado ** 4
    
    return (suma / total) - 3.0


def calcular_energia(imagen):
    """
    Calcula la energia de la imagen manualmente.
    
    Formula: energia = sum(pixel^2) / N
    
    Mide la concentracion de intensidades. Imagenes con valores
    concentrados en pocos niveles tienen mayor energia.
    """
    alto, ancho = imagen.shape
    total = alto * ancho
    suma = 0.0
    
    for i in range(alto):
        for j in range(ancho):
            valor = float(imagen[i, j]) / 255.0  # Normalizar a [0, 1]
            suma += valor * valor
    
    return suma / total


def calcular_entropia(imagen):
    """
    Calcula la entropia de Shannon de la imagen manualmente.
    
    Formula: entropia = -sum(p(i) * log2(p(i)))
    
    donde p(i) es la probabilidad de cada nivel de intensidad.
    Mide la cantidad de informacion (desorden) en la imagen.
    Mayor entropia = mayor variabilidad de intensidades.
    """
    histograma = calcular_histograma(imagen)
    alto, ancho = imagen.shape
    total = alto * ancho
    entropia = 0.0
    
    for frecuencia in histograma:
        if frecuencia > 0:
            probabilidad = frecuencia / total
            entropia -= probabilidad * math.log2(probabilidad)
    
    return entropia


def calcular_contraste(imagen):
    """
    Calcula el contraste de la imagen (diferencia max - min).
    
    Un contraste alto indica buena separacion entre zonas claras y oscuras.
    """
    minimo = 255
    maximo = 0
    alto, ancho = imagen.shape
    
    for i in range(alto):
        for j in range(ancho):
            val = int(imagen[i, j])
            if val < minimo:
                minimo = val
            if val > maximo:
                maximo = val
    
    return maximo - minimo


def extraer_caracteristicas(imagen):
    """
    Extrae el vector completo de caracteristicas de una imagen.
    
    Retorna un diccionario con todos los descriptores calculados.
    Estos descriptores son utiles para clasificacion porque:
    - Media: nivel general de brillo (separa imagenes claras de oscuras)
    - Desv. Estandar: variabilidad de intensidades (texturas vs zonas uniformes)
    - Skewness: asimetria de la distribucion (concentracion en claros u oscuros)
    - Kurtosis: forma de la distribucion (picos vs distribucion plana)
    - Energia: concentracion de valores (imagenes binarias vs graduales)
    - Entropia: complejidad de la imagen (ruido vs patron definido)
    - Contraste: rango dinamico de la imagen
    """
    caracteristicas = {
        'media': round(calcular_media(imagen), 4),
        'desviacion_estandar': round(calcular_desviacion_estandar(imagen), 4),
        'skewness': round(calcular_skewness(imagen), 4),
        'kurtosis': round(calcular_kurtosis(imagen), 4),
        'energia': round(calcular_energia(imagen), 6),
        'entropia': round(calcular_entropia(imagen), 4),
        'contraste': calcular_contraste(imagen),
        'histograma': calcular_histograma(imagen)
    }
    
    return caracteristicas


def formato_caracteristicas(caracteristicas):
    """
    Formatea las caracteristicas para mostrarlas en texto.
    Excluye el histograma (se muestra como grafica).
    """
    lineas = []
    nombres = {
        'media': 'Media de intensidad',
        'desviacion_estandar': 'Desviacion estandar',
        'skewness': 'Asimetria (Skewness)',
        'kurtosis': 'Curtosis (Kurtosis)',
        'energia': 'Energia',
        'entropia': 'Entropia (bits)',
        'contraste': 'Contraste (max-min)'
    }
    
    for clave, nombre in nombres.items():
        valor = caracteristicas[clave]
        lineas.append(f"{nombre}: {valor}")
    
    return "\n".join(lineas)
