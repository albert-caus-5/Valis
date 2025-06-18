import cv2
import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from register import Register
from image_cropper import ImageCropper
from transfering_annotations import AnnotationTransferer

class CellImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cell Image Processing")

        # Title label
        self.title_label = tk.Label(root, text="Cell Image Processing", font=("Helvetica", 16))
        self.title_label.pack(pady=10)

        # Options for registration
        self.registration_label = tk.Label(root, text="Image Registration")
        self.registration_label.pack(pady=5)
        
        self.registration_var = tk.StringVar(value="normal")
        self.registration_normal = tk.Radiobutton(root, text="Normal", variable=self.registration_var, value="normal")
        self.registration_normal.pack(anchor="w")
        self.registration_hd = tk.Radiobutton(root, text="HD", variable=self.registration_var, value="hd")
        self.registration_hd.pack(anchor="w")

        self.register_button = tk.Button(root, text="Register Images", command=self.register_images)
        self.register_button.pack(pady=10)

        # Transfer annotations
        self.transfer_button = tk.Button(root, text="Transfer Annotations", command=self.transfer_annotations)
        self.transfer_button.pack(pady=10)

        # Crop image
        self.crop_button = tk.Button(root, text="Crop Image", command=self.crop_image)
        self.crop_button.pack(pady=10)
        
        # Crop and Transfer simultaneously
        self.crop_and_transfer_button = tk.Button(root, text="Crop and Transfer", command=self.crop_and_transfer)
        self.crop_and_transfer_button.pack(pady=10)

    def register_images(self):
        # Función para registrar las imágenes
        
        # Preguntar cuántas imágenes se van a registrar
        num_images = simpledialog.askinteger("Número de Imágenes", "¿Cuántas imágenes deseas registrar?", minvalue=2)

        if not num_images or num_images < 2:
            messagebox.showwarning("Advertencia", "Debe registrar al menos 2 imágenes.")
            return

        # Solicitar el path de cada imagen
        image_paths = []
        for i in range(num_images):
            image_path = filedialog.askopenfilename(title=f"Selecciona la imagen {i + 1} para registrar",
                                                    filetypes=[
                                                        ("All supported files", "*.ome.tiff *.mrxs *.tif *.tiff"),  # Combined filter
                                                        ("OME-TIFF files", "*.ome.tiff"),
                                                        ("MRXS files", "*.mrxs"),
                                                        ("TIF files", "*.tif"),
                                                        ("TIFF files", "*.tiff")
                                                    ])
            if image_path:
                image_paths.append(image_path)
            else:
                messagebox.showwarning("Advertencia", "Debes seleccionar una imagen.")
                return

        # Solicitar el directorio donde se guardarán los resultados
        save_dir = filedialog.askdirectory(title="Selecciona el Directorio para guardar los resultados de la Registración")

        if not save_dir:
            messagebox.showwarning("Advertencia", "Debe seleccionar un directorio para guardar los resultados.")
            return

        registration_type = self.registration_var.get()
        messagebox.showinfo("Registro", f"Registrando imágenes con calidad {registration_type}, guardando resultados en {save_dir}")
        
        # Modificar la clase Register para aceptar una lista de paths de imágenes en lugar de un directorio
        register = Register(image_paths, save_dir, registration_type)
            
        if registration_type == "normal":
            register.registration()
        elif registration_type == "hd":
            register.registration_hd()
        else:
            messagebox.showerror("Error", "¡Tipo de registro inválido seleccionado!")



    def transfer_annotations(self):
        # Función para transferir anotaciones
        
        # Preguntar cuántas imágenes se van a registrar
        num_images = simpledialog.askinteger("Número de Imágenes", "¿Cuántas imágenes deseas registrar?", minvalue=2)

        if not num_images or num_images < 2:
            messagebox.showwarning("Advertencia", "Debe registrar al menos 2 imágenes.")
            return

        # Solicitar el path de cada imagen
        image_paths = []
        for i in range(num_images):
            if i == 0:
                # La primera imagen será la source_image_path
                image_path = filedialog.askopenfilename(
                    title=f"Selecciona la imagen {i + 1} (Fuente con Anotaciones)",
                    filetypes=[
                        ("All supported files", "*.ome.tiff *.mrxs *.tif *.tiff"),  # Combined filter
                        ("OME-TIFF files", "*.ome.tiff"),
                        ("MRXS files", "*.mrxs"),
                        ("TIF files", "*.tif"),
                        ("TIFF files", "*.tiff")
                    ]
                )
                if image_path:
                    source_image_path = image_path
                else:
                    messagebox.showwarning("Advertencia", "Debes seleccionar la imagen fuente.")
                    return
            else:
                # Las siguientes imágenes serán las target_image_paths
                image_path = filedialog.askopenfilename(
                    title=f"Selecciona la imagen {i + 1} (Objetivo para las Anotaciones)",
                    filetypes=[
                        ("All supported files", "*.ome.tiff *.mrxs *.tif *.tiff"),  # Combined filter
                        ("OME-TIFF files", "*.ome.tiff"),
                        ("MRXS files", "*.mrxs"),
                        ("TIF files", "*.tif"),
                        ("TIFF files", "*.tiff")
                    ]
                )
                if image_path:
                    image_paths.append(image_path)
                else:
                    messagebox.showwarning("Advertencia", "Debes seleccionar una imagen objetivo.")
                    return

        # Convertir la lista de imagenes objetivos a target_image_paths
        target_image_paths = image_paths

        # Solicitar el directorio donde se guardarán los resultados
        save_dir = filedialog.askdirectory(title="Selecciona el Directorio para guardar los resultados de la Registración")

        if not save_dir:
            messagebox.showwarning("Advertencia", "Debe seleccionar un directorio para guardar los resultados.")
            return
        
        # Solicitar el path del archivo GeoJSON con anotaciones
        geojson_path = filedialog.askopenfilename(
            title="Selecciona el Archivo GeoJSON con Anotaciones",
            filetypes=[("GeoJSON Files", "*.geojson")]
        )
        
        # Solicitar el directorio donde se guardarán las nuevas anotaciones GeoJSON
        save_geojson_path = filedialog.askdirectory(
            title="Selecciona el Directorio para Guardar las Nuevas Anotaciones GeoJSON"
        )
        
        registration_type = self.registration_var.get()
        
        if not source_image_path or not target_image_paths or not geojson_path or not save_geojson_path:
            messagebox.showwarning("Advertencia", "¡Debe seleccionar todas las rutas!")
            return
        
        # Modificar la clase AnnotationTransferer para aceptar una lista de paths de imágenes en lugar de un directorio
        transferer = AnnotationTransferer(
            save_dir,
            source_image_path,
            target_image_paths,
            geojson_path,
            save_geojson_path,
            registration_type
        )
        
        transferer.transfer_annotations()

        
    def crop_image(self):
        # Function to crop the image
        image_path = filedialog.askopenfilename(
            title="Select Image to Crop",
            filetypes=[
                ("All supported files", "*.ome.tiff *.mrxs *.tif *.tiff"),  # Combined filter
                ("OME-TIFF files", "*.ome.tiff"),
                ("MRXS files", "*.mrxs"),
                ("TIFF files", "*.tif"),
                ("TIFF files", "*.tiff")
            ]
        )
        save_image_path = filedialog.askdirectory(title="Select Directory to Save Cropped Image")
        geojson_path = filedialog.askopenfilename(title="Select GeoJSON File with Cropping Region", filetypes=[("GeoJSON Files", "*.geojson")])
        
        if not image_path or not save_image_path or not geojson_path:
            messagebox.showwarning("Warning", "All paths must be selected!")
            return
        
        # Check if there are annotations to consider
        annotations_geojson_path = None
        if messagebox.askyesno("Annotations", "Are there annotations in the region to crop?"):
            annotations_geojson_path = filedialog.askopenfilename(title="Select GeoJSON File with Annotations", filetypes=[("GeoJSON Files", "*.geojson")])
            
            if not annotations_geojson_path:
                messagebox.showwarning("Warning", "You selected Yes for annotations, but no path was provided. Continuing without annotations.")
            else:
                messagebox.showinfo("Annotations Selected", f"Annotations will be loaded from {annotations_geojson_path}")
    
         
        messagebox.showinfo("Crop Image", f"Cropping {image_path} with region from {geojson_path}, saving to {save_image_path}")
        
        #### Create ImageCropper instance with the paths
        cropper = ImageCropper(image_path, save_image_path, geojson_path, annotations_geojson_path)
        cropper.crop_and_save_image()

    def crop_and_transfer(self):
        
        # Preguntar cuántas imágenes se van a registrar
        num_images = simpledialog.askinteger("Número de Imágenes", "¿Cuántas imágenes deseas registrar?", minvalue=2)

        if not num_images or num_images < 2:
            messagebox.showwarning("Advertencia", "Debe registrar al menos 2 imágenes.")
            return

        # Solicitar el path de cada imagen
        image_paths = []
        for i in range(num_images):
            if i == 0:
                # La primera imagen será la source_image_path
                image_path = filedialog.askopenfilename(
                    title=f"Selecciona la imagen {i + 1} (Fuente con Anotaciones)",
                    filetypes=[
                        ("All supported files", "*.ome.tiff *.mrxs *.tif *.tiff"),  # Combined filter
                        ("OME-TIFF files", "*.ome.tiff"),
                        ("MRXS files", "*.mrxs"),
                        ("TIF files", "*.tif"),
                        ("TIFF files", "*.tiff")
                    ]
                )
                if image_path:
                    source_image_path = image_path
                else:
                    messagebox.showwarning("Advertencia", "Debes seleccionar la imagen fuente.")
                    return
            else:
                # Las siguientes imágenes serán las target_image_paths
                image_path = filedialog.askopenfilename(
                    title=f"Selecciona la imagen {i + 1} (Objetivo para las Anotaciones)",
                    filetypes=[
                        ("All supported files", "*.ome.tiff *.mrxs *.tif *.tiff"),  # Combined filter
                        ("OME-TIFF files", "*.ome.tiff"),
                        ("MRXS files", "*.mrxs"),
                        ("TIF files", "*.tif"),
                        ("TIFF files", "*.tiff")
                    ]
                )
                if image_path:
                    image_paths.append(image_path)
                else:
                    messagebox.showwarning("Advertencia", "Debes seleccionar una imagen objetivo.")
                    return

        # Convertir la lista de imagenes objetivos a target_image_paths
        target_image_paths = image_paths

        # Solicitar el directorio donde se guardarán los resultados
        save_dir = filedialog.askdirectory(title="Selecciona el Directorio para guardar los resultados de la Registración")

        if not save_dir:
            messagebox.showwarning("Advertencia", "Debe seleccionar un directorio para guardar los resultados.")
            return
        
        # Solicitar el path del archivo GeoJSON con anotaciones
        geojson_path = filedialog.askopenfilename(
            title="Selecciona el Archivo GeoJSON con Anotaciones",
            filetypes=[("GeoJSON Files", "*.geojson")]
        )
        
        # Solicitar el directorio donde se guardarán las nuevas anotaciones GeoJSON
        save_geojson_path = filedialog.askdirectory(
            title="Selecciona el Directorio para Guardar las Nuevas Anotaciones GeoJSON"
        )
        
        # Solicitar el path del archivo GeoJSON para recortar la source image
        geojson_source_roi_path = filedialog.askopenfilename(
            title="Selecciona el Archivo GeoJSON la region a recortar para la Source Image",
            filetypes=[("GeoJSON Files", "*.geojson")]
        )
        
        geojson_target_roi_paths  = []
        for i in range(num_images - 1):
            # Solicitar el path del archivo GeoJSON con anotaciones
            geojson_target_path = filedialog.askopenfilename(
                title=f"Selecciona el Archivo GeoJSON con la Región de Recorte para la Imagen Objetivo {i + 1}",
                filetypes=[("GeoJSON Files", "*.geojson")]
            )
        
            if geojson_target_path:
                geojson_target_roi_paths.append(geojson_target_path)
            else:
                messagebox.showwarning("Advertencia", "Debe seleccionar el archivo GeoJSON con la región de recorte para todas las imágenes objetivo.")
                return
        
        registration_type = self.registration_var.get()
        
        if not source_image_path or not target_image_paths or not geojson_path or not save_geojson_path:
            messagebox.showwarning("Advertencia", "¡Debe seleccionar todas las rutas!")
            return
        
        # Modificar la clase AnnotationTransferer para aceptar una lista de paths de imágenes en lugar de un directorio
        transferer = AnnotationTransferer(
            save_dir,
            source_image_path,
            target_image_paths,
            geojson_path,
            save_geojson_path,
            registration_type
        )
        transferer.cropNtransfer_annotations(geojson_source_roi_path, geojson_target_roi_paths)

        messagebox.showinfo("Proceso Completo", "El proceso de recorte y transferencia de anotaciones se ha completado.")

    
if __name__ == "__main__":
    root = tk.Tk()
    app = CellImageApp(root)
    root.mainloop()
