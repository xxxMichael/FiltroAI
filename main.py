"""
Programa principal - GUI interactiva para procesamiento de imagenes.
Filtrado de ruido sal y pimienta en dominio espacial y frecuencia.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import threading
import os

from preprocessing import convertir_a_grises, agregar_ruido_sal_pimienta
from spatial_filters import filtro_media, filtro_mediana, filtro_moda
from frequency_filters import filtrar_en_frecuencia, obtener_magnitud_espectro
from metrics import calcular_mse, calcular_psnr
from feature_extraction import extraer_caracteristicas, formato_caracteristicas, calcular_histograma

# Tamano fijo para las miniaturas (evita que crezcan)
THUMB_MAX_W = 280
THUMB_MAX_H = 250


class AplicacionFiltros:
    def __init__(self, root):
        self.root = root
        self.root.title("Filtrado de Imagenes - IA")
        self.root.state('zoomed')

        # Estado de la aplicacion
        self.imagen_original = None
        self.imagen_gris = None
        self.imagen_ruidosa = None
        self.imagen_filtrada = None
        self.imagen_frecuencia = None
        self.resultados_metricas = []

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

        # --- Seccion: Frecuencia (solo diametro) ---
        sec5 = ttk.LabelFrame(panel_ctrl, text="4. Filtro en Frecuencia")
        sec5.pack(fill=tk.X, padx=5, pady=3)
        ttk.Label(sec5, text="Tipo:").pack(padx=5, anchor=tk.W)
        self.var_freq = tk.StringVar(value="Pasa-Bajo (circulo)")
        ttk.Combobox(sec5, textvariable=self.var_freq, values=["Pasa-Bajo (circulo)", "Pasa-Alto (circulo)"], state='readonly').pack(fill=tk.X, padx=5)
        ttk.Label(sec5, text="Diametro del circulo (px):").pack(padx=5, anchor=tk.W)
        self.var_diametro = tk.IntVar(value=60)
        ttk.Scale(sec5, from_=10, to=300, variable=self.var_diametro, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5)
        self.lbl_diam = ttk.Label(sec5, text="60 px")
        self.lbl_diam.pack(padx=5)
        self.var_diametro.trace_add('write', lambda *a: self.lbl_diam.config(text=f"{self.var_diametro.get()} px"))
        ttk.Button(sec5, text="Aplicar Filtro Frecuencia", command=self._aplicar_filtro_frecuencia).pack(fill=tk.X, padx=5, pady=3)

        # --- Seccion: Analisis ---
        sec6 = ttk.LabelFrame(panel_ctrl, text="5. Analisis")
        sec6.pack(fill=tk.X, padx=5, pady=3)
        ttk.Button(sec6, text="Extraer Caracteristicas", command=self._extraer_caracteristicas).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(sec6, text="Comparar Metricas", command=self._comparar_metricas).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(sec6, text="Mostrar Histograma", command=self._mostrar_histograma).pack(fill=tk.X, padx=5, pady=2)

        # Panel central - Imagenes (2x3 grid, sin bordes)
        panel_imgs = ttk.Frame(main)
        panel_imgs.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.frames_img = {}
        self.labels_img = {}
        self.fotos = {}
        nombres = [
            ("original", "Original (Grises)"), ("ruidosa", "Con Ruido"),
            ("filtrada", "Filtrada (Espacial)"), ("frecuencia", "Filtrada (Frecuencia)"),
            ("espectro", "Espectro FFT"), ("mascara", "Mascara Frecuencia")
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

        # Panel derecho - Resultados
        panel_res = ttk.LabelFrame(main, text="Resultados", width=300)
        panel_res.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        panel_res.pack_propagate(False)
        self.texto_resultados = tk.Text(panel_res, wrap=tk.WORD, font=('Consolas', 9), state=tk.DISABLED)
        scroll = ttk.Scrollbar(panel_res, command=self.texto_resultados.yview)
        self.texto_resultados.config(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.texto_resultados.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)

        # Barra de estado
        self.barra_estado = ttk.Label(self.root, text="Listo", relief=tk.SUNKEN, anchor=tk.W)
        self.barra_estado.pack(side=tk.BOTTOM, fill=tk.X)

    def _escribir_resultado(self, texto, limpiar=False):
        self.texto_resultados.config(state=tk.NORMAL)
        if limpiar:
            self.texto_resultados.delete('1.0', tk.END)
        self.texto_resultados.insert(tk.END, texto + "\n")
        self.texto_resultados.see(tk.END)
        self.texto_resultados.config(state=tk.DISABLED)

    def _estado(self, texto):
        self.barra_estado.config(text=texto)
        self.root.update_idletasks()

    def _mostrar_imagen(self, clave, imagen_np):
        """Muestra una imagen numpy en el label correspondiente con tamano fijo."""
        lbl = self.labels_img[clave]
        img_pil = Image.fromarray(imagen_np, mode='L')

        # Escalar a tamano fijo para que no crezcan los frames
        ratio = min(THUMB_MAX_W / img_pil.width, THUMB_MAX_H / img_pil.height)
        if ratio > 1:
            ratio = 1  # No agrandar imagenes pequenas
        nuevo_w = max(int(img_pil.width * ratio), 1)
        nuevo_h = max(int(img_pil.height * ratio), 1)
        img_pil = img_pil.resize((nuevo_w, nuevo_h), Image.NEAREST)

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
            self.imagen_original = np.array(img)
            self._estado("Convirtiendo a escala de grises...")

            if len(self.imagen_original.shape) == 3:
                if self.imagen_original.shape[2] == 4:
                    self.imagen_original = self.imagen_original[:, :, :3]
                self.imagen_gris = convertir_a_grises(self.imagen_original)
            else:
                self.imagen_gris = self.imagen_original.copy()

            self._mostrar_imagen("original", self.imagen_gris)
            alto, ancho = self.imagen_gris.shape
            self.lbl_info.config(text=f"{os.path.basename(ruta)}\n{ancho}x{alto} px")
            self._escribir_resultado(f"Imagen cargada: {os.path.basename(ruta)} ({ancho}x{alto})", limpiar=True)
            self._estado("Imagen cargada correctamente")
            self.imagen_ruidosa = None
            self.imagen_filtrada = None
            self.imagen_frecuencia = None
            self.resultados_metricas = []
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar la imagen:\n{e}")

    def _aplicar_ruido(self):
        if self.imagen_gris is None:
            messagebox.showwarning("Aviso", "Primero carga una imagen.")
            return
        porcentaje = self.var_ruido.get()
        self._estado(f"Aplicando ruido sal y pimienta ({porcentaje}%)...")
        self.imagen_ruidosa = agregar_ruido_sal_pimienta(self.imagen_gris, porcentaje)
        self._mostrar_imagen("ruidosa", self.imagen_ruidosa)
        mse = calcular_mse(self.imagen_gris, self.imagen_ruidosa)
        psnr = calcular_psnr(self.imagen_gris, self.imagen_ruidosa)
        self._escribir_resultado(f"\n--- Ruido S&P ({porcentaje}%) ---\nMSE: {mse:.2f}\nPSNR: {psnr:.2f} dB")
        self._estado("Ruido aplicado")

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
        self._mostrar_imagen("filtrada", self.imagen_filtrada)
        mse = calcular_mse(self.imagen_gris, self.imagen_filtrada)
        psnr = calcular_psnr(self.imagen_gris, self.imagen_filtrada)
        self._escribir_resultado(f"\n--- Filtro {tipo} ({tamano}x{tamano}) ---\nMSE: {mse:.2f}\nPSNR: {psnr:.2f} dB")
        self.resultados_metricas.append({'filtro': f"{tipo} {tamano}x{tamano}", 'mse': mse, 'psnr': psnr})
        self._estado(f"Filtro {tipo} aplicado correctamente")

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
            # Convertir mascara a uint8 para mostrarla
            mascara_visual = (mascara * 255).astype(np.uint8)
            self.root.after(0, lambda: self._frecuencia_listo(resultado, magnitud, mascara_visual, tipo, diametro))

        threading.Thread(target=tarea, daemon=True).start()

    def _frecuencia_listo(self, resultado, magnitud, mascara_visual, tipo, diametro):
        self.imagen_frecuencia = resultado
        self._mostrar_imagen("frecuencia", self.imagen_frecuencia)
        self._mostrar_imagen("espectro", magnitud)
        self._mostrar_imagen("mascara", mascara_visual)
        mse = calcular_mse(self.imagen_gris, self.imagen_frecuencia)
        psnr = calcular_psnr(self.imagen_gris, self.imagen_frecuencia)
        self._escribir_resultado(f"\n--- {tipo} (diametro={diametro}px) ---\nMSE: {mse:.2f}\nPSNR: {psnr:.2f} dB")
        self.resultados_metricas.append({'filtro': f"{tipo} d={diametro}", 'mse': mse, 'psnr': psnr})
        self._estado(f"Filtro frecuencia aplicado")

    def _extraer_caracteristicas(self):
        imagenes = {
            'Original (Grises)': self.imagen_gris,
            'Con Ruido': self.imagen_ruidosa,
            'Filtrada (Espacial)': self.imagen_filtrada,
            'Filtrada (Frecuencia)': self.imagen_frecuencia
        }
        disponibles = {k: v for k, v in imagenes.items() if v is not None}
        if not disponibles:
            messagebox.showwarning("Aviso", "No hay imagenes para analizar.")
            return
        self._estado("Extrayendo caracteristicas...")
        self._escribir_resultado("\n" + "=" * 40 + "\nEXTRACCION DE CARACTERISTICAS\n" + "=" * 40)
        for nombre, img in disponibles.items():
            caract = extraer_caracteristicas(img)
            self._escribir_resultado(f"\n--- {nombre} ---\n{formato_caracteristicas(caract)}")
        self._estado("Caracteristicas extraidas")

    def _comparar_metricas(self):
        if not self.resultados_metricas:
            messagebox.showwarning("Aviso", "Aplica al menos un filtro primero.")
            return
        self._escribir_resultado("\n" + "=" * 40 + "\nCOMPARACION DE METRICAS\n" + "=" * 40)
        self._escribir_resultado(f"{'Filtro':<30} {'MSE':>10} {'PSNR (dB)':>12}")
        self._escribir_resultado("-" * 54)
        for r in self.resultados_metricas:
            self._escribir_resultado(f"{r['filtro']:<30} {r['mse']:>10.2f} {r['psnr']:>12.2f}")
        mejor = min(self.resultados_metricas, key=lambda x: x['mse'])
        self._escribir_resultado(f"\nMejor filtro (menor MSE): {mejor['filtro']}")

    def _mostrar_histograma(self):
        imagenes = {
            'Original': self.imagen_gris,
            'Con Ruido': self.imagen_ruidosa,
            'Filtrada Espacial': self.imagen_filtrada,
            'Filtrada Frecuencia': self.imagen_frecuencia
        }
        disponibles = {k: v for k, v in imagenes.items() if v is not None}
        if not disponibles:
            messagebox.showwarning("Aviso", "No hay imagenes para mostrar histograma.")
            return
        fig, axes = plt.subplots(1, len(disponibles), figsize=(5 * len(disponibles), 4))
        if len(disponibles) == 1:
            axes = [axes]
        for ax, (nombre, img) in zip(axes, disponibles.items()):
            hist = calcular_histograma(img)
            ax.bar(range(256), hist, color='gray', width=1)
            ax.set_title(nombre)
            ax.set_xlabel('Intensidad')
            ax.set_ylabel('Frecuencia')
        fig.suptitle('Histogramas de Intensidad')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    root = tk.Tk()
    app = AplicacionFiltros(root)
    root.mainloop()
