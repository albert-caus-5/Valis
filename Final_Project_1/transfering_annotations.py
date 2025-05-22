import json
import os
import numpy as np
import pyvips
from tkinter import messagebox
from register import Register
from image_cropper import ImageCropper

class AnnotationTransferer:
    def __init__(self, save_dir, source_image_path, target_image_paths, geojson_path, save_geojson_path, registration_type):
        #self.images_dir = images_dir
        self.image_paths = [source_image_path] + target_image_paths
        self.save_dir = save_dir
        self.source_image_path = source_image_path
        self.target_image_paths = target_image_paths
        self.geojson_path = geojson_path
        self.save_geojson_path = save_geojson_path
        self.registration_type = registration_type
        
        # Define the auxiliary crop directory
        self.aux_crop_dir = os.path.join(self.save_dir, "aux_crop")

        # Check if the auxiliary crop directory exists, and create it if it doesn't
        if not os.path.exists(self.aux_crop_dir):
            os.makedirs(self.aux_crop_dir)

    def extract_coordinates_from_geojson(self, geojson_path):
        
        """
        Extracts coordinates from a GeoJSON file.

        Args:
            geojson_path (str): Path to the GeoJSON file.

        Returns:
            numpy.ndarray: Array of coordinates.
        """
        
        # Leer el archivo GeoJSON
        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)
        
        # Lista para almacenar todas las coordenadas
        all_coordinates = []

        # Iterar sobre todas las características
        for feature in geojson_data['features']:
            geometry = feature['geometry']
            
            # Manejar diferentes tipos de geometrías
            if geometry['type'] == 'Polygon':
                # Para cada polígono, se puede tener un anillo externo y posiblemente varios internos
                for polygon in geometry['coordinates']:
                    all_coordinates.extend(polygon)
            elif geometry['type'] == 'MultiPolygon':
                # Cada MultiPolygon es una lista de polígonos, y cada polígono es una lista de anillos
                for multipolygon in geometry['coordinates']:
                    for polygon in multipolygon:
                        all_coordinates.extend(polygon)
            else:
                raise ValueError(f"Tipo de geometría no soportada: {geometry['type']}")

        # Convertir la lista de coordenadas a un array numpy
        xy = np.array(all_coordinates)
        
        return xy

    def adjust_coords_source(self, coords, image):
        
        # Get the properties of the bounding box
        bounds_x = image.get('openslide.bounds-x')
        bounds_y = image.get('openslide.bounds-y')
        bounds_width = image.get('openslide.bounds-width')
        bounds_height = image.get('openslide.bounds-height')

        # Adjust the coordinates using the offset of the bounding box
        coords_adjust = []
        for coord in coords:
            x_ajustado = coord[0] + int(bounds_x)
            y_ajustado = coord[1] + int(bounds_y)
            coords_adjust.append([x_ajustado, y_ajustado])
                
        return coords_adjust
    
    def adjust_coords_target(self, coords, image):
        
        # Get the properties of the bounding box
        bounds_x = image.get('openslide.bounds-x')
        bounds_y = image.get('openslide.bounds-y')
        bounds_width = image.get('openslide.bounds-width')
        bounds_height = image.get('openslide.bounds-height')

        # Adjust the coordinates using the offset of the bounding box
        coords_adjust = []
        for coord in coords:
            x_ajustado = coord[0] - int(bounds_x)
            y_ajustado = coord[1] - int(bounds_y)
            coords_adjust.append([x_ajustado, y_ajustado])
                
        return coords_adjust 
    
    def save_coordinates_to_geojson(self, coordinates, target_image_path, geojson_path_transference):
        """
        Saves coordinates to a GeoJSON file, maintaining its structure and ensuring no empty polygons.

        Args:
            coordinates (numpy.ndarray or list): Array or list of coordinates.
            original_geojson_path (str): Path to the original GeoJSON file.
            output_geojson_path (str): Path to the output GeoJSON file.
        """

        # Load the original GeoJSON file to maintain its structure
        with open(geojson_path_transference, 'r') as f:
            geojson_data = json.load(f)

        # Iterate over features and update coordinates
        coord_index = 0
        for feature in geojson_data['features']:
            geometry = feature['geometry']
            
            if geometry['type'] == 'Polygon':
                # Updated list of polygons to avoid empty polygons
                new_polygons = []
                for polygon in geometry['coordinates']:
                    num_coords = len(polygon)
                    
                    if coord_index + num_coords <= len(coordinates):
                        if isinstance(coordinates, np.ndarray):
                            new_polygon = coordinates[coord_index:coord_index + num_coords].tolist()
                        else:
                            new_polygon = coordinates[coord_index:coord_index + num_coords]
                        
                        # Only add non-empty polygons
                        if new_polygon:
                            new_polygons.append(new_polygon)
                    
                    coord_index += num_coords

                geometry['coordinates'] = new_polygons

            elif geometry['type'] == 'MultiPolygon':
                # Updated list of multipolygons to avoid empty polygons
                new_multipolygons = []
                for multipolygon in geometry['coordinates']:
                    new_multipolygon = []
                    for polygon in multipolygon:
                        num_coords = len(polygon)
                        
                        if coord_index + num_coords <= len(coordinates):
                            if isinstance(coordinates, np.ndarray):
                                new_polygon = coordinates[coord_index:coord_index + num_coords].tolist()
                            else:
                                new_polygon = coordinates[coord_index:coord_index + num_coords]
                            
                            # Only add non-empty polygons
                            if new_polygon:
                                new_multipolygon.append(new_polygon)
                        
                        coord_index += num_coords
                    
                    if new_multipolygon:
                        new_multipolygons.append(new_multipolygon)

                geometry['coordinates'] = new_multipolygons

        # Obtain the file name from the target_image_path
        target_image_filename = os.path.basename(target_image_path)
        # Change the file extension to .geojson
        geojson_filename = os.path.splitext(target_image_filename)[0] + ".geojson"
        # Construct the full path for the GeoJSON file
        geojson_path = os.path.join(self.save_geojson_path, geojson_filename)

        # Save the new GeoJSON to a file with pretty formatting
        with open(geojson_path, 'w') as f:
            json.dump(geojson_data, f, indent=4)

    def transfer_annotations(self):
        """
        Performs the registration and transfers annotations from the source image to the target image.
        """
        
        # Perform registration
        register = Register(self.image_paths, self.save_dir, self.registration_type)
        if (self.registration_type == "normal"):
            annotation_source_slide, target_slides = register.registration(transfer=True, source_image_path=self.source_image_path, target_image_paths=self.target_image_paths)
        elif (self.registration_type == "hd"):
            annotation_source_slide, target_slides = register.registration_hd(transfer=True, source_image_path=self.source_image_path, target_image_paths=self.target_image_paths)
        else:
            messagebox.showerror("Error", "Invalid registration type selected!")
        
        # Read your annotations
        annotation_pt_xy = self.extract_coordinates_from_geojson(self.geojson_path)
        
        # If it is a .mrxs file, the coordinates must be adjusted due to the bounding box
        if self.source_image_path.endswith('.mrxs'):
            image = pyvips.Image.new_from_file(self.source_image_path)
            annotation_pt_xy = self.adjust_coords_source(annotation_pt_xy, image)

        for target_slide, target_path in zip(target_slides, self.target_image_paths):
            # Warp the annotations to the target slide
            warped_coordinates = annotation_source_slide.warp_xy_from_to(annotation_pt_xy, target_slide)
            
            # Check if the target image is .mrxs and adjust coordinates if necessary
            if target_path.endswith('.mrxs'):
                target_image = pyvips.Image.new_from_file(target_path)
                warped_coordinates = self.adjust_coords_target(warped_coordinates, target_image)
            
        for target_path in self.target_image_paths:
            # Save the new coordinates to a new GeoJSON file
            self.save_coordinates_to_geojson(warped_coordinates, target_path, self.geojson_path)

    def cropNtransfer_annotations(self, geojson_path_roi_source, geojson_path_roi_target):
        """
        Recorta las imágenes usando los GeoJSON proporcionados y transfiere las anotaciones entre ellas.
        """
        # Define the auxiliary crop directory
        aux_crop_dir = os.path.join(self.save_dir, "aux_crop")

        # Check if the auxiliary crop directory exists, and create it if it doesn't
        if not os.path.exists(aux_crop_dir):
            os.makedirs(aux_crop_dir)

        print(self.target_image_paths)
        for target_path in self.target_image_paths:
            print(target_path)

        # Recortar la imagen de origen y obtener las coordenadas de recorte
        source_cropper = ImageCropper(self.source_image_path, aux_crop_dir, geojson_path_roi_source, annotations_geojson_path=self.geojson_path)
        cropped_source_image_path, crop_x_source, crop_y_source, _, _ = source_cropper.crop_and_save_image(return_crop_coords=True)

        # Recortar las imágenes de destino y obtener las coordenadas de recorte
        cropped_target_image_paths = []
        target_crop_coords = []
        for target_path, geojson_path in zip(self.target_image_paths, geojson_path_roi_target):
            target_cropper = ImageCropper(str(target_path), aux_crop_dir, geojson_path)
            cropped_target_image_path, crop_x_target, crop_y_target, _, _ = target_cropper.crop_and_save_image(return_crop_coords=True)
            cropped_target_image_paths.append(cropped_target_image_path)
            target_crop_coords.append((crop_x_target, crop_y_target))

        '''
        print("--------------")
        print(cropped_source_image_path)
        print(cropped_target_image_paths)
        print("--------------")
        '''
        
        # Realizar el registro con las imágenes recortadas
        register = Register([cropped_source_image_path] + cropped_target_image_paths, self.save_dir, self.registration_type)
        if self.registration_type == "normal":
            annotation_source_slide, target_slides = register.registration(transfer=True, source_image_path=cropped_source_image_path, target_image_paths=cropped_target_image_paths)
        elif self.registration_type == "hd":
            annotation_source_slide, target_slides = register.registration_hd(transfer=True, source_image_path=cropped_source_image_path, target_image_paths=cropped_target_image_paths)
        else:
            messagebox.showerror("Error", "¡Tipo de registro inválido seleccionado!")

        # Obtener el directorio donde se encuentra el archivo original
        base_dir = os.path.dirname(self.geojson_path)
        # Nombre del archivo GeoJSON sin la extensión
        filename_without_ext = os.path.splitext(os.path.basename(self.geojson_path))[0]
        # Crear el nuevo nombre del archivo con "_crop" añadido antes de la extensión
        cropped_filename = f"{filename_without_ext}_crop.geojson"
        # Crear el nuevo directorio "cropped" dentro del directorio base
        cropped_dir = os.path.join(base_dir, "cropped")
        # Asegurarse de que el directorio "cropped" exista
        os.makedirs(cropped_dir, exist_ok=True)
        # Construir la ruta completa para el archivo GeoJSON recortado
        cropped_geojson_path = os.path.join(cropped_dir, cropped_filename)
        
        '''
        print("-------------------")
        print(cropped_geojson_path)
        print(cropped_source_image_path)
        print("-------------------")
        '''
        
        # Leer las anotaciones de la imagen de origen recortada
        annotation_pt_xy = self.extract_coordinates_from_geojson(cropped_geojson_path)
        
        '''
        print("-------------------")
        print(annotation_pt_xy)
        print("-------------------")       
        '''
        
        # Ajustar coordenadas para la imagen recortada de origen si es necesario (caso .mrxs)
        if cropped_source_image_path.endswith('.mrxs'):
            print("dentro de if de mrxs")
            image = pyvips.Image.new_from_file(cropped_source_image_path)
            annotation_pt_xy = self.adjust_coords_source(annotation_pt_xy, image)

        # Warpar las anotaciones y ajustar para las imágenes de destino originales
        for target_slide, target_path, (crop_x_target, crop_y_target), original_target_path in zip(target_slides, cropped_target_image_paths, target_crop_coords, self.target_image_paths):
            # Warpar las anotaciones a la imagen de destino recortada
            warped_coordinates = annotation_source_slide.warp_xy_from_to(annotation_pt_xy, target_slide)
            
            # Ajustar coordenadas para la imagen recortada de destino si es necesario (caso .mrxs)
            if target_path.endswith('.mrxs'):
                print("dentro de if de mrxs")
                target_image = pyvips.Image.new_from_file(target_path)
                warped_coordinates = self.adjust_coords_target(warped_coordinates, target_image)

            # Define the "cropped" subdirectory
            annotations_output_dir2 = os.path.join(os.path.dirname(original_target_path), "croppedHE")
            # Define the "cropped" subdirectory
            annotations_output_dir1 = os.path.join(os.path.dirname(original_target_path), "croppedHER2")
            
            '''
            print("-------------------")
            print(warped_coordinates)
            print("-------------------")   
            '''
            
            #Guardar los pasos de las anotaciones
            self.save_coordinates_to_geojson(warped_coordinates, annotations_output_dir1, cropped_geojson_path)
            self.save_coordinates_to_geojson(annotation_pt_xy, annotations_output_dir2, cropped_geojson_path)

            # Check if the target image is .mrxs and adjust coordinates if necessary
            if original_target_path.endswith('.mrxs'):
                target_image = pyvips.Image.new_from_file(original_target_path)
                warped_coordinates = self.adjust_coords_target(warped_coordinates, target_image)
            
            # Ajustar las coordenadas según el recorte
            final_adjusted_coordinates = []
            for coord in warped_coordinates:
                final_adjusted_coordinates.append([coord[0] + crop_x_target, coord[1] + crop_y_target])
            
            # Guardar las coordenadas ajustadas en un nuevo archivo GeoJSON
            self.save_coordinates_to_geojson(final_adjusted_coordinates, original_target_path, cropped_geojson_path)
