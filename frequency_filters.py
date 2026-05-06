"""
Filtros en el dominio de la frecuencia.
Transformada de Fourier (FFT) y filtros ideales circulares (pasa-bajo y pasa-alto).
El unico parametro es el diametro del circulo central.
"""

import numpy as np
import math


def aplicar_fft(imagen):
    """
    Aplica la Transformada Rapida de Fourier (FFT) a una imagen.
    
    Proceso:
    1. FFT 2D de la imagen
    2. Centrar las frecuencias bajas (shift)
    
    Parametros:
        imagen: numpy array 2D (escala de grises)
    
    Retorna:
        espectro: Representacion en frecuencia (compleja, centrada)
    """
    espectro = np.fft.fft2(imagen.astype(np.float64))
    espectro_centrado = np.fft.fftshift(espectro)
    return espectro_centrado


def aplicar_ifft(espectro_centrado):
    """
    Aplica la Transformada Inversa de Fourier para regresar al dominio espacial.
    
    Parametros:
        espectro_centrado: numpy array 2D complejo (centrado)
    
    Retorna:
        imagen: numpy array 2D (valores reales)
    """
    espectro = np.fft.ifftshift(espectro_centrado)
    imagen = np.fft.ifft2(espectro)
    imagen_real = np.abs(imagen)
    return imagen_real


def obtener_magnitud_espectro(espectro_centrado):
    """
    Calcula la magnitud del espectro en escala logaritmica para visualizacion.
    
    Formula: magnitud = log(1 + |F(u,v)|)
    
    Parametros:
        espectro_centrado: numpy array 2D complejo
    
    Retorna:
        magnitud: numpy array 2D (float64) normalizada a [0, 255]
    """
    alto, ancho = espectro_centrado.shape
    magnitud = np.zeros((alto, ancho), dtype=np.float64)
    
    for i in range(alto):
        for j in range(ancho):
            magnitud[i, j] = math.log(1 + abs(espectro_centrado[i, j]))
    
    # Normalizar a [0, 255]
    minimo = magnitud.min()
    maximo = magnitud.max()
    if maximo - minimo > 0:
        for i in range(alto):
            for j in range(ancho):
                magnitud[i, j] = (magnitud[i, j] - minimo) / (maximo - minimo) * 255.0
    
    return magnitud.astype(np.uint8)


def crear_filtro_ideal_circular_pb(alto, ancho, diametro):
    """
    Crea una mascara circular ideal pasa-bajo.
    
    Es un circulo blanco (valor 1) en el centro del espectro con el
    diametro especificado. Todo lo que esta fuera del circulo es 0.
    
    - Dentro del circulo: deja pasar las frecuencias bajas (zonas suaves)
    - Fuera del circulo: bloquea las frecuencias altas (ruido, bordes)
    
    Parametros:
        alto: altura de la imagen
        ancho: ancho de la imagen
        diametro: diametro del circulo en pixeles
    
    Retorna:
        mascara: numpy array 2D (float64) con valores 0 o 1
    """
    mascara = np.zeros((alto, ancho), dtype=np.float64)
    centro_y = alto // 2
    centro_x = ancho // 2
    radio = diametro / 2.0
    
    for u in range(alto):
        for v in range(ancho):
            distancia = math.sqrt((u - centro_y) ** 2 + (v - centro_x) ** 2)
            if distancia <= radio:
                mascara[u, v] = 1.0
            else:
                mascara[u, v] = 0.0
    
    return mascara


def crear_filtro_ideal_circular_pa(alto, ancho, diametro):
    """
    Crea una mascara circular ideal pasa-alto.
    
    Es el inverso del pasa-bajo: circulo negro (valor 0) en el centro
    y todo lo demas en blanco (valor 1).
    
    - Dentro del circulo: bloquea frecuencias bajas
    - Fuera del circulo: deja pasar frecuencias altas (bordes, detalles)
    
    Parametros:
        alto: altura de la imagen
        ancho: ancho de la imagen
        diametro: diametro del circulo en pixeles
    
    Retorna:
        mascara: numpy array 2D (float64) con valores 0 o 1
    """
    mascara_pb = crear_filtro_ideal_circular_pb(alto, ancho, diametro)
    mascara_pa = np.zeros((alto, ancho), dtype=np.float64)
    
    for u in range(alto):
        for v in range(ancho):
            mascara_pa[u, v] = 1.0 - mascara_pb[u, v]
    
    return mascara_pa


def filtrar_en_frecuencia(imagen, tipo_filtro='ideal_pb', diametro=60):
    """
    Pipeline completo de filtrado en el dominio de la frecuencia.
    
    Proceso:
    1. Aplicar FFT a la imagen
    2. Crear mascara circular con el diametro dado
    3. Multiplicar espectro por mascara
    4. Aplicar FFT inversa
    
    Parametros:
        imagen: numpy array 2D (escala de grises, uint8)
        tipo_filtro: 'ideal_pb' (pasa-bajo) o 'ideal_pa' (pasa-alto)
        diametro: diametro del circulo central en pixeles
    
    Retorna:
        imagen_filtrada: numpy array 2D uint8
        espectro_original: espectro centrado (para visualizacion)
        mascara: mascara de frecuencia usada (para visualizacion)
        espectro_filtrado: espectro despues de aplicar la mascara
    """
    alto, ancho = imagen.shape
    
    # 1. Transformada de Fourier
    espectro = aplicar_fft(imagen)
    
    # 2. Crear mascara circular
    if tipo_filtro == 'ideal_pb':
        mascara = crear_filtro_ideal_circular_pb(alto, ancho, diametro)
    else:  # ideal_pa
        mascara = crear_filtro_ideal_circular_pa(alto, ancho, diametro)
    
    # 3. Multiplicar espectro por mascara
    espectro_filtrado = np.zeros_like(espectro, dtype=np.complex128)
    for u in range(alto):
        for v in range(ancho):
            espectro_filtrado[u, v] = espectro[u, v] * mascara[u, v]
    
    # 4. Transformada inversa
    imagen_resultado = aplicar_ifft(espectro_filtrado)
    
    # Normalizar resultado a [0, 255]
    minimo = imagen_resultado.min()
    maximo = imagen_resultado.max()
    if maximo - minimo > 0:
        imagen_resultado = (imagen_resultado - minimo) / (maximo - minimo) * 255.0
    
    imagen_uint8 = np.zeros((alto, ancho), dtype=np.uint8)
    for i in range(alto):
        for j in range(ancho):
            val = int(round(imagen_resultado[i, j]))
            if val < 0:
                val = 0
            elif val > 255:
                val = 255
            imagen_uint8[i, j] = val
    
    return imagen_uint8, espectro, mascara, espectro_filtrado
