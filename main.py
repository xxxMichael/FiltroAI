"""
Programa principal - GUI interactiva para procesamiento de imagenes.
Filtrado de ruido sal y pimienta en dominio espacial y frecuencia.

Pipeline:
  RGB -> Grises -> Normalizar histograma -> Binarizar -> Ruido -> Filtrar
  Al final se muestran las imagenes convertidas de vuelta a RGB.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import threading
import os

from preprocessing import (convertir_a_grises, normalizar_histograma,
                           binarizar, agregar_ruido_sal_pimienta, gris_a_rgb)
from spatial_filters import filtro_media, filtro_mediana, filtro_moda
from frequency_filters import filtrar_en_frecuencia, obtener_magnitud_espectro
from metrics import calcular_mse, calcular_psnr

# Tamano de respaldo si el frame aun no tiene dimensiones
THUMB_FALLBACK_W = 400
THUMB_FALLBACK_H = 350


class AplicacionFiltros:
    def __init__(self, root):
        self.root = root
        self.root.title("Filtrado de Imagenes - IA")
        self.root.state('zoomed')

        # Estado de la aplicacion
        self.imagen_original_rgb = None
        self.imagen_gris = None
        self.imagen_normalizada = None
        self.imagen_binarizada = None
        self.imagen_ruidosa = None
        self.imagen_filtrada = None
        self.imagen_frecuencia = None
        self.display_w = 0
        self.display_h = 0

        self._construir_interfaz()

    def _construir_interfaz(self):
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Panel izquierdo - Controles
        panel_ctrl = ttk.LabelFrame(main, text="Controles", width=280)
        panel_ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        panel_ctrl.pack_propagate(False)

        # --- Seccion: Cargar Imagen ---
        sec1 = ttk.LabelFrame(panel_ctrl, text="1. Imagen")
        sec1.pack(fill=tk.X, padx=5, pady=3)
        ttk.Button(sec1, text="Cargar Imagen", command=self._cargar_imagen).pack(fill=tk.X, padx=5, pady=3)
        self.lbl_info = ttk.Label(sec1, text="Sin imagen", wraplength=250)
        self.lbl_info.pack(padx=5, pady=2)

        # --- Seccion: Ruido ---
        sec2 = ttk.LabelFrame(panel_ctrl, text="2. Ruido Sal y Pimienta")
        sec2.pack(fill=tk.X, padx=5, pady=3)
        ttk.Label(sec2, text="Nivel de ruido (%):").pack(padx=5, anchor=tk.W)
        self.var_ruido = tk.IntVar(value=10)
        ttk.Scale(sec2, from_=1, to=50, variable=self.var_ruido, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5)
        self.lbl_ruido = ttk.Label(sec2, text="10%")
        self.lbl_ruido.pack(padx=5)
        self.var_ruido.trace_add('write', lambda *a: self.lbl_ruido.config(text=f"{self.var_ruido.get()}%"))
        ttk.Button(sec2, text="Aplicar Ruido", command=self._aplicar_ruido).pack(fill=tk.X, padx=5, pady=3)

        # --- Seccion: Filtros Espaciales ---
        sec3 = ttk.LabelFrame(panel_ctrl, text="3. Filtros Espaciales")
        sec3.pack(fill=tk.X, padx=5, pady=3)
        ttk.Label(sec3, text="Tipo de filtro:").pack(padx=5, anchor=tk.W)
        self.var_filtro = tk.StringVar(value="Mediana")
        ttk.Combobox(sec3, textvariable=self.var_filtro, values=["Media", "Mediana", "Moda"], state='readonly').pack(fill=tk.X, padx=5)
        ttk.Label(sec3, text="Tamano del kernel:").pack(padx=5, anchor=tk.W)
        self.var_kernel = tk.StringVar(value="3x3")
        ttk.Combobox(sec3, textvariable=self.var_kernel, values=["3x3", "5x5", "7x7"], state='readonly').pack(fill=tk.X, padx=5)
        ttk.Button(sec3, text="Aplicar Filtro Espacial", command=self._aplicar_filtro_espacial).pack(fill=tk.X, padx=5, pady=3)

        # --- Seccion: Filtro en Frecuencia ---
        sec4 = ttk.LabelFrame(panel_ctrl, text="4. Filtro en Frecuencia")
        sec4.pack(fill=tk.X, padx=5, pady=3)
        ttk.Label(sec4, text="Tipo:").pack(padx=5, anchor=tk.W)
        self.var_freq = tk.StringVar(value="Pasa-Bajo")
        ttk.Combobox(sec4, textvariable=self.var_freq, values=["Pasa-Bajo", "Pasa-Alto"], state='readonly').pack(fill=tk.X, padx=5)
        ttk.Label(sec4, text="Diametro del circulo (px):").pack(padx=5, anchor=tk.W)
        self.var_diametro = tk.IntVar(value=60)
        ttk.Scale(sec4, from_=10, to=300, variable=self.var_diametro, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5)
        self.lbl_diam = ttk.Label(sec4, text="60 px")
        self.lbl_diam.pack(padx=5)
        self.var_diametro.trace_add('write', lambda *a: self.lbl_diam.config(text=f"{self.var_diametro.get()} px"))
        ttk.Button(sec4, text="Aplicar Filtro Frecuencia", command=self._aplicar_filtro_frecuencia).pack(fill=tk.X, padx=5, pady=3)

        # Panel central - Imagenes (2x3 grid)
        panel_imgs = ttk.Frame(main)
        panel_imgs.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.frames_img = {}
        self.labels_img = {}
        self.fotos = {}
        nombres = [
            ("original",   "1. Original (RGB)"),
            ("ruidosa",    "2. Con Ruido (RGB)"),
            ("filtrada",   "3. Filtrada Espacial (RGB)"),
            ("frecuencia", "4. Filtrada Frecuencia (RGB)"),
            ("espectro",   "5. Espectro FFT"),
            ("mascara",    "6. Mascara Frecuencia")
        ]
        for idx, (clave, titulo) in enumerate(nombres):
            fila = idx // 3
            col = idx % 3
            frame = ttk.LabelFrame(panel_imgs, text=titulo)
            frame.grid(row=fila, column=col, padx=3, pady=3, sticky='nsew')
            lbl = ttk.Label(frame, text="Sin imagen", anchor=tk.CENTER)
            lbl.pack(expand=True, fill=tk.BOTH, padx=2, pady=2)
            self.frames_img[clave] = frame
            self.labels_img[clave] = lbl
        for i in range(2):
            panel_imgs.rowconfigure(i, weight=1)
        for j in range(3):
            panel_imgs.columnconfigure(j, weight=1)

        # Barra de estado
        self.barra_estado = ttk.Label(self.root, text="Listo", relief=tk.SUNKEN, anchor=tk.W)
        self.barra_estado.pack(side=tk.BOTTOM, fill=tk.X)

    def _estado(self, texto):
        self.barra_estado.config(text=texto)
        self.root.update_idletasks()

    def _calcular_tamano_display(self, img_w, img_h):
        """Calcula el tamano de display UNA sola vez al cargar la imagen."""
        frame = self.frames_img["original"]
        frame.update_idletasks()
        fw = frame.winfo_width() - 16
        fh = frame.winfo_height() - 36
        if fw < 50:
            fw = THUMB_FALLBACK_W
        if fh < 50:
            fh = THUMB_FALLBACK_H
        ratio = min(fw / img_w, fh / img_h)
        self.display_w = max(int(img_w * ratio), 1)
        self.display_h = max(int(img_h * ratio), 1)

    def _mostrar_imagen_gris(self, clave, imagen_np):
        """Muestra una imagen en escala de grises en el label."""
        lbl = self.labels_img[clave]
        img_pil = Image.fromarray(imagen_np, mode='L')
        img_pil = img_pil.resize((self.display_w, self.display_h), Image.LANCZOS)
        foto = ImageTk.PhotoImage(img_pil)
        lbl.config(image=foto, text="")
        self.fotos[clave] = foto

    def _mostrar_imagen_rgb(self, clave, imagen_np_rgb):
        """Muestra una imagen RGB en el label."""
        lbl = self.labels_img[clave]
        img_pil = Image.fromarray(imagen_np_rgb, mode='RGB')
        img_pil = img_pil.resize((self.display_w, self.display_h), Image.LANCZOS)
        foto = ImageTk.PhotoImage(img_pil)
        lbl.config(image=foto, text="")
        self.fotos[clave] = foto

    def _cargar_imagen(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Imagenes", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("Todos", "*.*")]
        )
        if not ruta:
            return
        try:
            img = Image.open(ruta)
            self.imagen_original_rgb = np.array(img)

            # Si tiene canal alpha, quedarse con RGB
            if len(self.imagen_original_rgb.shape) == 3 and self.imagen_original_rgb.shape[2] == 4:
                self.imagen_original_rgb = self.imagen_original_rgb[:, :, :3]

            # Si es escala de grises, convertir a RGB
            if len(self.imagen_original_rgb.shape) == 2:
                self.imagen_original_rgb = gris_a_rgb(self.imagen_original_rgb)

            self._estado("Preprocesando: RGB -> Grises -> Normalizar -> Binarizar...")

            # Calcular tamano de display UNA sola vez
            alto_img, ancho_img = self.imagen_original_rgb.shape[0], self.imagen_original_rgb.shape[1]
            self._calcular_tamano_display(ancho_img, alto_img)

            # 1. Mostrar original RGB
            self._mostrar_imagen_rgb("original", self.imagen_original_rgb)

            # 2. Convertir a escala de grises (background)
            self.imagen_gris = convertir_a_grises(self.imagen_original_rgb)

            # 3. Normalizar histograma (background)
            self.imagen_normalizada = normalizar_histograma(self.imagen_gris)

            # 4. Binarizar (background)
            self.imagen_binarizada = binarizar(self.imagen_normalizada)

            alto, ancho = self.imagen_gris.shape
            self.lbl_info.config(text=f"{os.path.basename(ruta)}\n{ancho}x{alto} px")
            self._estado("Imagen cargada y preprocesada")

            # Resetear
            self.imagen_ruidosa = None
            self.imagen_filtrada = None
            self.imagen_frecuencia = None
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar la imagen:\n{e}")

    def _aplicar_ruido(self):
        if self.imagen_binarizada is None:
            messagebox.showwarning("Aviso", "Primero carga una imagen.")
            return
        porcentaje = self.var_ruido.get()
        self._estado(f"Aplicando ruido sal y pimienta ({porcentaje}%)...")
        self.imagen_ruidosa = agregar_ruido_sal_pimienta(self.imagen_binarizada, porcentaje)
        self._mostrar_imagen_rgb("ruidosa", gris_a_rgb(self.imagen_ruidosa))
        mse = calcular_mse(self.imagen_binarizada, self.imagen_ruidosa)
        psnr = calcular_psnr(self.imagen_binarizada, self.imagen_ruidosa)
        self._estado(f"Ruido aplicado — MSE: {mse:.2f} | PSNR: {psnr:.2f} dB")

    def _aplicar_filtro_espacial(self):
        if self.imagen_ruidosa is None:
            messagebox.showwarning("Aviso", "Primero aplica ruido a la imagen.")
            return
        tipo = self.var_filtro.get()
        tamano = int(self.var_kernel.get().split('x')[0])
        self._estado(f"Aplicando filtro {tipo} ({tamano}x{tamano})... Esto puede tardar.")
        self.root.update_idletasks()

        def tarea():
            if tipo == "Media":
                resultado = filtro_media(self.imagen_ruidosa, tamano)
            elif tipo == "Mediana":
                resultado = filtro_mediana(self.imagen_ruidosa, tamano)
            else:
                resultado = filtro_moda(self.imagen_ruidosa, tamano)
            self.root.after(0, lambda: self._filtro_espacial_listo(resultado, tipo, tamano))

        threading.Thread(target=tarea, daemon=True).start()

    def _filtro_espacial_listo(self, resultado, tipo, tamano):
        self.imagen_filtrada = resultado
        self._mostrar_imagen_rgb("filtrada", gris_a_rgb(self.imagen_filtrada))
        mse = calcular_mse(self.imagen_binarizada, self.imagen_filtrada)
        psnr = calcular_psnr(self.imagen_binarizada, self.imagen_filtrada)
        self._estado(f"Filtro {tipo} ({tamano}x{tamano}) — MSE: {mse:.2f} | PSNR: {psnr:.2f} dB")

    def _aplicar_filtro_frecuencia(self):
        if self.imagen_ruidosa is None:
            messagebox.showwarning("Aviso", "Primero aplica ruido a la imagen.")
            return
        tipo = self.var_freq.get()
        diametro = self.var_diametro.get()
        tipo_interno = 'ideal_pb' if 'Bajo' in tipo else 'ideal_pa'
        self._estado(f"Aplicando {tipo} (diametro={diametro}px)... Esto puede tardar.")
        self.root.update_idletasks()

        def tarea():
            resultado, espectro_orig, mascara, espectro_filt = filtrar_en_frecuencia(
                self.imagen_ruidosa, tipo_interno, diametro
            )
            magnitud = obtener_magnitud_espectro(espectro_orig)
            mascara_visual = (mascara * 255).astype(np.uint8)
            self.root.after(0, lambda: self._frecuencia_listo(resultado, magnitud, mascara_visual, tipo, diametro))

        threading.Thread(target=tarea, daemon=True).start()

    def _frecuencia_listo(self, resultado, magnitud, mascara_visual, tipo, diametro):
        self.imagen_frecuencia = resultado
        self._mostrar_imagen_rgb("frecuencia", gris_a_rgb(self.imagen_frecuencia))
        self._mostrar_imagen_gris("espectro", magnitud)
        self._mostrar_imagen_gris("mascara", mascara_visual)
        mse = calcular_mse(self.imagen_binarizada, self.imagen_frecuencia)
        psnr = calcular_psnr(self.imagen_binarizada, self.imagen_frecuencia)
        self._estado(f"{tipo} (d={diametro}px) — MSE: {mse:.2f} | PSNR: {psnr:.2f} dB")


if __name__ == '__main__':
    root = tk.Tk()
    app = AplicacionFiltros(root)
    root.mainloop()
