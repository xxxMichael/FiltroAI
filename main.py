"""
Programa principal - Interfaz grafica para procesamiento de imagenes.
Pipeline: RGB -> Grises -> Normalizar -> Binarizar -> Ruido -> Filtrar
Permite alternar la visualizacion entre escala de grises y RGB en cada panel.
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

THUMB_FALLBACK_W = 400
THUMB_FALLBACK_H = 350


class AplicacionFiltros:
    def __init__(self, root):
        self.root = root
        self.root.title("Filtrado de Imagenes - IA")
        self.root.state('zoomed')

        self.imagen_original_rgb = None
        self.imagen_gris = None
        self.imagen_normalizada = None
        self.imagen_binarizada = None
        self.imagen_ruidosa = None
        self.imagen_filtrada = None
        self.imagen_frecuencia = None
        # Versiones RGB (procesadas canal por canal)
        self.ruidosa_rgb = None
        self.filtrada_rgb = None
        self.frecuencia_rgb = None
        self.display_w = 0
        self.display_h = 0
        # Modo de vista: 'grises' o 'rgb'
        self.vista = {'ruidosa': 'grises', 'filtrada': 'grises', 'frecuencia': 'grises'}
        self.btns_toggle = {}

        self._construir_interfaz()

    def _construir_interfaz(self):
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        panel_ctrl = ttk.LabelFrame(main, text="Controles", width=280)
        panel_ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        panel_ctrl.pack_propagate(False)

        # 1. Imagen
        sec1 = ttk.LabelFrame(panel_ctrl, text="1. Imagen")
        sec1.pack(fill=tk.X, padx=5, pady=3)
        ttk.Button(sec1, text="Cargar Imagen", command=self._cargar_imagen).pack(fill=tk.X, padx=5, pady=3)
        self.lbl_info = ttk.Label(sec1, text="Sin imagen", wraplength=250)
        self.lbl_info.pack(padx=5, pady=2)

        # 2. Ruido
        sec2 = ttk.LabelFrame(panel_ctrl, text="2. Ruido Sal y Pimienta")
        sec2.pack(fill=tk.X, padx=5, pady=3)
        ttk.Label(sec2, text="Nivel de ruido (%):").pack(padx=5, anchor=tk.W)
        self.var_ruido = tk.IntVar(value=10)
        ttk.Scale(sec2, from_=1, to=50, variable=self.var_ruido, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5)
        self.lbl_ruido = ttk.Label(sec2, text="10%")
        self.lbl_ruido.pack(padx=5)
        self.var_ruido.trace_add('write', lambda *a: self.lbl_ruido.config(text=f"{self.var_ruido.get()}%"))
        ttk.Button(sec2, text="Aplicar Ruido", command=self._aplicar_ruido).pack(fill=tk.X, padx=5, pady=3)

        # 3. Filtros Espaciales
        sec3 = ttk.LabelFrame(panel_ctrl, text="3. Filtros Espaciales")
        sec3.pack(fill=tk.X, padx=5, pady=3)
        ttk.Label(sec3, text="Tipo de filtro:").pack(padx=5, anchor=tk.W)
        self.var_filtro = tk.StringVar(value="Mediana")
        ttk.Combobox(sec3, textvariable=self.var_filtro, values=["Media", "Mediana", "Moda"], state='readonly').pack(fill=tk.X, padx=5)
        ttk.Label(sec3, text="Tamano del kernel:").pack(padx=5, anchor=tk.W)
        self.var_kernel = tk.StringVar(value="3x3")
        ttk.Combobox(sec3, textvariable=self.var_kernel, values=["3x3", "5x5", "7x7"], state='readonly').pack(fill=tk.X, padx=5)
        ttk.Button(sec3, text="Aplicar Filtro Espacial", command=self._aplicar_filtro_espacial).pack(fill=tk.X, padx=5, pady=3)

        # 4. Filtro en Frecuencia
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

        # Panel central - Imagenes (2x3)
        panel_imgs = ttk.Frame(main)
        panel_imgs.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.frames_img = {}
        self.labels_img = {}
        self.fotos = {}
        nombres = [
            ("original",   "1. Original "),
            ("ruidosa",    "2. Con Ruido"),
            ("filtrada",   "3. Filtrada Espacial"),
            ("frecuencia", "4. Filtrada Frecuencia"),
            ("espectro",   "5. Espectro FFT"),
            ("mascara",    "6. Mascara Frecuencia")
        ]
        toggleables = {"ruidosa", "filtrada", "frecuencia"}
        for idx, (clave, titulo) in enumerate(nombres):
            fila, col = idx // 3, idx % 3
            frame = ttk.LabelFrame(panel_imgs, text=titulo)
            frame.grid(row=fila, column=col, padx=3, pady=3, sticky='nsew')
            lbl = ttk.Label(frame, text="Sin imagen", anchor=tk.CENTER)
            lbl.pack(expand=True, fill=tk.BOTH, padx=2, pady=2)
            self.frames_img[clave] = frame
            self.labels_img[clave] = lbl
            if clave in toggleables:
                btn = ttk.Button(frame, text="Ver RGB",
                                 command=lambda c=clave: self._toggle_vista(c))
                btn.pack(pady=(0, 3))
                self.btns_toggle[clave] = btn
        for i in range(2):
            panel_imgs.rowconfigure(i, weight=1)
        for j in range(3):
            panel_imgs.columnconfigure(j, weight=1)

        self.barra_estado = ttk.Label(self.root, text="Listo", relief=tk.SUNKEN, anchor=tk.W)
        self.barra_estado.pack(side=tk.BOTTOM, fill=tk.X)

    def _estado(self, texto):
        self.barra_estado.config(text=texto)
        self.root.update_idletasks()

    def _calcular_tamano_display(self, img_w, img_h):
        frame = self.frames_img["original"]
        frame.update_idletasks()
        fw = frame.winfo_width() - 16
        fh = frame.winfo_height() - 36
        if fw < 50: fw = THUMB_FALLBACK_W
        if fh < 50: fh = THUMB_FALLBACK_H
        ratio = min(fw / img_w, fh / img_h)
        self.display_w = max(int(img_w * ratio), 1)
        self.display_h = max(int(img_h * ratio), 1)

    def _mostrar_imagen_gris(self, clave, imagen_np):
        lbl = self.labels_img[clave]
        img_pil = Image.fromarray(imagen_np, mode='L')
        img_pil = img_pil.resize((self.display_w, self.display_h), Image.LANCZOS)
        foto = ImageTk.PhotoImage(img_pil)
        lbl.config(image=foto, text="")
        self.fotos[clave] = foto

    def _mostrar_imagen_rgb(self, clave, imagen_np_rgb):
        lbl = self.labels_img[clave]
        img_pil = Image.fromarray(imagen_np_rgb, mode='RGB')
        img_pil = img_pil.resize((self.display_w, self.display_h), Image.LANCZOS)
        foto = ImageTk.PhotoImage(img_pil)
        lbl.config(image=foto, text="")
        self.fotos[clave] = foto

    def _mostrar_panel(self, clave):
        """Muestra la imagen correcta segun el modo de vista actual."""
        modo = self.vista.get(clave, 'grises')
        if modo == 'rgb':
            rgb_map = {'ruidosa': self.ruidosa_rgb,
                       'filtrada': self.filtrada_rgb,
                       'frecuencia': self.frecuencia_rgb}
            img_rgb = rgb_map.get(clave)
            if img_rgb is not None:
                self._mostrar_imagen_rgb(clave, img_rgb)
                return
        # Grises (default)
        gris_map = {'ruidosa': self.imagen_ruidosa,
                    'filtrada': self.imagen_filtrada,
                    'frecuencia': self.imagen_frecuencia}
        img_gris = gris_map.get(clave)
        if img_gris is not None:
            self._mostrar_imagen_rgb(clave, gris_a_rgb(img_gris))

    def _toggle_vista(self, clave):
        """Alterna entre vista grises y RGB para un panel."""
        rgb_map = {'ruidosa': self.ruidosa_rgb,
                   'filtrada': self.filtrada_rgb,
                   'frecuencia': self.frecuencia_rgb}
        if self.vista[clave] == 'grises':
            if rgb_map[clave] is None:
                messagebox.showinfo("Aviso", "Aun no hay version RGB. Aplica la operacion primero.")
                return
            self.vista[clave] = 'rgb'
            self.btns_toggle[clave].config(text="Ver Grises")
        else:
            self.vista[clave] = 'grises'
            self.btns_toggle[clave].config(text="Ver RGB")
        self._mostrar_panel(clave)

    def _aplicar_filtro_canal(self, canal, tipo, tamano):
        """Aplica un filtro espacial a un solo canal."""
        if tipo == "Media":
            return filtro_media(canal, tamano)
        elif tipo == "Mediana":
            return filtro_mediana(canal, tamano)
        else:
            return filtro_moda(canal, tamano)

    # --- Cargar imagen ---
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
            if len(self.imagen_original_rgb.shape) == 3 and self.imagen_original_rgb.shape[2] == 4:
                self.imagen_original_rgb = self.imagen_original_rgb[:, :, :3]
            if len(self.imagen_original_rgb.shape) == 2:
                self.imagen_original_rgb = gris_a_rgb(self.imagen_original_rgb)

            self._estado("Preprocesando...")
            alto_img, ancho_img = self.imagen_original_rgb.shape[0], self.imagen_original_rgb.shape[1]
            self._calcular_tamano_display(ancho_img, alto_img)
            self._mostrar_imagen_rgb("original", self.imagen_original_rgb)

            self.imagen_gris = convertir_a_grises(self.imagen_original_rgb)
            self.imagen_normalizada = normalizar_histograma(self.imagen_gris)
            self.imagen_binarizada = binarizar(self.imagen_normalizada)

            self.lbl_info.config(text=f"{os.path.basename(ruta)}\n{ancho_img}x{alto_img} px")
            self._estado("Imagen cargada y preprocesada")
            # Resetear
            self.imagen_ruidosa = None
            self.imagen_filtrada = None
            self.imagen_frecuencia = None
            self.ruidosa_rgb = None
            self.filtrada_rgb = None
            self.frecuencia_rgb = None
            for k in self.vista:
                self.vista[k] = 'grises'
                self.btns_toggle[k].config(text="Ver RGB")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar la imagen:\n{e}")

    # --- Ruido ---
    def _aplicar_ruido(self):
        if self.imagen_binarizada is None:
            messagebox.showwarning("Aviso", "Primero carga una imagen.")
            return
        porcentaje = self.var_ruido.get()
        self._estado(f"Aplicando ruido ({porcentaje}%)...")
        # Grises: ruido sobre binarizada
        self.imagen_ruidosa = agregar_ruido_sal_pimienta(self.imagen_binarizada, porcentaje)
        # RGB: ruido sobre cada canal del original
        alto, ancho = self.imagen_original_rgb.shape[0], self.imagen_original_rgb.shape[1]
        self.ruidosa_rgb = np.zeros((alto, ancho, 3), dtype=np.uint8)
        for c in range(3):
            self.ruidosa_rgb[:, :, c] = agregar_ruido_sal_pimienta(
                self.imagen_original_rgb[:, :, c], porcentaje)
        # Resetear filtros (nuevo ruido invalida los anteriores)
        self.imagen_filtrada = None
        self.imagen_frecuencia = None
        self.filtrada_rgb = None
        self.frecuencia_rgb = None
        self.vista['ruidosa'] = 'grises'
        self.btns_toggle['ruidosa'].config(text="Ver RGB")
        self._mostrar_panel("ruidosa")
        mse = calcular_mse(self.imagen_binarizada, self.imagen_ruidosa)
        psnr = calcular_psnr(self.imagen_binarizada, self.imagen_ruidosa)
        self._estado(f"Ruido aplicado — MSE: {mse:.2f} | PSNR: {psnr:.2f} dB")

    # --- Filtro Espacial ---
    def _aplicar_filtro_espacial(self):
        if self.imagen_ruidosa is None:
            messagebox.showwarning("Aviso", "Primero aplica ruido a la imagen.")
            return
        tipo = self.var_filtro.get()
        tamano = int(self.var_kernel.get().split('x')[0])
        self._estado(f"Aplicando filtro {tipo} ({tamano}x{tamano})... Esto puede tardar.")
        self.root.update_idletasks()

        def tarea():
            # Grises
            resultado_gris = self._aplicar_filtro_canal(self.imagen_ruidosa, tipo, tamano)
            # RGB (canal por canal)
            alto, ancho = self.imagen_original_rgb.shape[0], self.imagen_original_rgb.shape[1]
            resultado_rgb = np.zeros((alto, ancho, 3), dtype=np.uint8)
            if self.ruidosa_rgb is not None:
                for c in range(3):
                    resultado_rgb[:, :, c] = self._aplicar_filtro_canal(
                        self.ruidosa_rgb[:, :, c], tipo, tamano)
            self.root.after(0, lambda: self._espacial_listo(resultado_gris, resultado_rgb, tipo, tamano))

        threading.Thread(target=tarea, daemon=True).start()

    def _espacial_listo(self, resultado_gris, resultado_rgb, tipo, tamano):
        self.imagen_filtrada = resultado_gris
        self.filtrada_rgb = resultado_rgb
        self.vista['filtrada'] = 'grises'
        self.btns_toggle['filtrada'].config(text="Ver RGB")
        self._mostrar_panel("filtrada")
        mse = calcular_mse(self.imagen_binarizada, self.imagen_filtrada)
        psnr = calcular_psnr(self.imagen_binarizada, self.imagen_filtrada)
        self._estado(f"Filtro {tipo} ({tamano}x{tamano}) — MSE: {mse:.2f} | PSNR: {psnr:.2f} dB")

    # --- Filtro Frecuencia ---
    def _aplicar_filtro_frecuencia(self):
        if self.imagen_ruidosa is None:
            messagebox.showwarning("Aviso", "Primero aplica ruido a la imagen.")
            return
        tipo = self.var_freq.get()
        diametro = self.var_diametro.get()
        tipo_interno = 'ideal_pb' if 'Bajo' in tipo else 'ideal_pa'
        self._estado(f"Aplicando {tipo} (d={diametro}px)... Esto puede tardar.")
        self.root.update_idletasks()

        def tarea():
            # Grises
            res_gris, esp_orig, mascara, _ = filtrar_en_frecuencia(
                self.imagen_ruidosa, tipo_interno, diametro)
            magnitud = obtener_magnitud_espectro(esp_orig)
            mascara_vis = (mascara * 255).astype(np.uint8)
            # RGB (canal por canal)
            alto, ancho = self.imagen_original_rgb.shape[0], self.imagen_original_rgb.shape[1]
            res_rgb = np.zeros((alto, ancho, 3), dtype=np.uint8)
            if self.ruidosa_rgb is not None:
                for c in range(3):
                    canal_filt, _, _, _ = filtrar_en_frecuencia(
                        self.ruidosa_rgb[:, :, c], tipo_interno, diametro)
                    res_rgb[:, :, c] = canal_filt
            self.root.after(0, lambda: self._frecuencia_listo(
                res_gris, res_rgb, magnitud, mascara_vis, tipo, diametro))

        threading.Thread(target=tarea, daemon=True).start()

    def _frecuencia_listo(self, res_gris, res_rgb, magnitud, mascara_vis, tipo, diametro):
        self.imagen_frecuencia = res_gris
        self.frecuencia_rgb = res_rgb
        self.vista['frecuencia'] = 'grises'
        self.btns_toggle['frecuencia'].config(text="Ver RGB")
        self._mostrar_panel("frecuencia")
        self._mostrar_imagen_gris("espectro", magnitud)
        self._mostrar_imagen_gris("mascara", mascara_vis)
        mse = calcular_mse(self.imagen_binarizada, self.imagen_frecuencia)
        psnr = calcular_psnr(self.imagen_binarizada, self.imagen_frecuencia)
        self._estado(f"{tipo} (d={diametro}px) — MSE: {mse:.2f} | PSNR: {psnr:.2f} dB")


if __name__ == '__main__':
    root = tk.Tk()
    app = AplicacionFiltros(root)
    root.mainloop()
