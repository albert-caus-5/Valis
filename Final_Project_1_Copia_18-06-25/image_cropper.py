import json
import pyvips
import openslide
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__))) # Calling my_valis instead of installed valis
from my_valis import slide_io 

class ImageCropper:
    def __init__(self, reference_image, output_dir, geojson_path, annotations_geojson_path = None, pyramid_level = 0):
        self.reference_image = reference_image
        self.output_dir = output_dir
        self.geojson_path = geojson_path
        self.pyramid_level = pyramid_level
        self.annotations_geojson_path = annotations_geojson_path

    def read_geojson_coords(self):
        """
        Obtains the coordinates of the first polygon from a GeoJSON file.

        Returns:
            list: List of coordinates (x, y) in pixels.
        """
        with open(self.geojson_path, 'r') as f:
            data = json.load(f)

        # Extract the coordinates of the first polygon
        features = data['features']
        if not features:
            raise ValueError("No features found in the GeoJSON.")

        geometry = features[0]['geometry']
        if geometry['type'] != 'Polygon':
            raise ValueError("The geometry is not a polygon.")

        coordinates = geometry['coordinates'][0]
        return coordinates

    def read_annotations_coords(self):
        """
        Reads the coordinates of annotations from the GeoJSON file.

        Returns:
            numpy.ndarray: Array of annotations with coordinates.
        """
        with open(self.annotations_geojson_path, 'r') as f:
            data = json.load(f)

        annotations = []

        # Directamente trabajar con NumPy arrays si las coordenadas son numéricas
        for feature in data['features']:
            geometry = feature['geometry']
            
            if geometry['type'] == 'Point':
                coords = np.array(geometry['coordinates'], dtype=float)
                annotations.append(coords)
            elif geometry['type'] == 'Polygon':
                coords = np.array(geometry['coordinates'][0], dtype=float)  # Solo el primer anillo
                annotations.append(coords)
            elif geometry['type'] == 'MultiPolygon':
                for polygon in geometry['coordinates']:
                    coords = np.array(polygon[0], dtype=float)  # Solo el primer anillo de cada polígono
                    annotations.append(coords)
            elif geometry['type'] == 'LineString':
                coords = np.array(geometry['coordinates'], dtype=float)
                annotations.append(coords)
            elif geometry['type'] == 'MultiLineString':
                for linestring in geometry['coordinates']:
                    coords = np.array(linestring, dtype=float)
                    annotations.append(coords)

        # Combinar todas las coordenadas en un solo array de NumPy
        annotations_array = np.array(annotations, dtype=object)

        return annotations_array

    def crop_image(self, image):
        """
        Crops the image using the coordinates from the GeoJSON.

        Args:
            image (pyvips.Image): Image to crop.

        Returns:
            pyvips.Image: Cropped image.
        """
        # Get the coordinates from the GeoJSON
        coords = self.read_geojson_coords()
        
        # If it is a .mrxs file, the coordinates must be adjusted due to the bounding box
        if self.reference_image.endswith('.mrxs'):
            coords = self.adjust_coords_mrxs(coords, image)

        # Crop coordinates
        x = coords[0][0]
        y = coords[0][1]
        width = coords[1][0] - x
        height = coords[2][1] - y
        
        #print(x, y, width, height)
        
        # Crop the image
        cropped_image = image.crop(x, y, width, height)

        return cropped_image, x, y, width, height
    
    def adjust_coords_mrxs(self, coords, image):
        """
        Adjusts the coordinates according to the image's bounding box.

        Args:
            coords (list): Original coordinates.
            image (pyvips.Image): Image object.

        Returns:
            list: Adjusted coordinates.
        """
        
        # Get the properties of the bounding box
        bounds_x = image.get('openslide.bounds-x')
        bounds_y = image.get('openslide.bounds-y')

        # Adjust the coordinates using the offset of the bounding box
        coords_adjust = []
        for coord in coords:
            x_ajustado = coord[0] + int(bounds_x)
            y_ajustado = coord[1] + int(bounds_y)
            coords_adjust.append([x_ajustado, y_ajustado])
                
        return coords_adjust
    
    def adjust_annotation_coords(self, annotations, crop_x, crop_y, bounds_x=0, bounds_y=0):
        """
        Adjusts annotation coordinates according to the crop region and bounding box.

        Args:
            annotations (list): Original annotation coordinates.
            crop_x (int): X coordinate of the top-left corner of the crop region.
            crop_y (int): Y coordinate of the top-left corner of the crop region.
            bounds_x (int, optional): X offset of the bounding box. Defaults to 0.
            bounds_y (int, optional): Y offset of the bounding box. Defaults to 0.

        Returns:
            list: Adjusted annotation coordinates.
        """
        adjusted_annotations = []
        
        # Print original annotations for debugging
        #print("Original annotations:", annotations)
        
        for annotation in annotations:
            if isinstance(annotation[0], (list, np.ndarray)):  # Polygon
                adjusted_coords = [[float(x) - float(crop_x) + float(bounds_x), float(y) - float(crop_y) + float(bounds_y)] for x, y in annotation]
            else:  # Point
                adjusted_coords = [float(annotation[0]) - float(crop_x) + float(bounds_x), float(annotation[1]) - float(crop_y) + float(bounds_y)]
            adjusted_annotations.append(adjusted_coords)
        
        # Print adjusted annotations for debugging
        #print("Adjusted annotations:", adjusted_annotations)

        return adjusted_annotations
        
    def save_annotations_geojson(self, adjusted_annotations, crop_x, crop_y, width, height):
        """
        Saves the adjusted annotations to a new GeoJSON file.

        Args:
            adjusted_annotations (list): Adjusted annotation coordinates.
            crop_x (int): X coordinate of the top-left corner of the crop region.
            crop_y (int): Y coordinate of the top-left corner of the crop region.
            width (int): Width of the cropped region.
            height (int): Height of the cropped region.
        """
        new_features = []
        for annotation in adjusted_annotations:
            if isinstance(annotation[0], list):  # Polygon
                # Ensure the polygon is closed by checking the first and last points
                if annotation[0] != annotation[-1]:
                    annotation.append(annotation[0])

                geometry = {
                    "type": "Polygon",
                    "coordinates": [annotation]
                }
            else:  # Point
                geometry = {
                    "type": "Point",
                    "coordinates": annotation
                }
            new_features.append({
                "type": "Feature",
                "geometry": geometry,
                "properties": {}
            })

        new_geojson = {
            "type": "FeatureCollection",
            "features": new_features
        }

        # Create output path for annotations
        annotations_basename = os.path.basename(self.annotations_geojson_path)
        annotations_name, annotations_ext = os.path.splitext(annotations_basename)
        annotations_filename = f"{annotations_name}_crop{annotations_ext}"
        
        # Define the "cropped" subdirectory
        annotations_output_dir = os.path.join(os.path.dirname(self.annotations_geojson_path), "cropped")
        
        # Create the "cropped" subdirectory if it doesn't exist
        if not os.path.exists(annotations_output_dir):
            os.makedirs(annotations_output_dir)

        annotations_output_path = os.path.join(annotations_output_dir, annotations_filename)
        
        # Save to GeoJSON file
        with open(annotations_output_path, 'w') as f:
            json.dump(new_geojson, f, indent=4)

    def crop_and_save_image(self, return_crop_coords=False):
        """
        Crops the image using the GeoJSON and saves the result as an OME-TIFF file.
        Optionally returns the cropping coordinates.

        Args:
            return_crop_coords (bool): If True, returns the coordinates of the crop region.

        Returns:
            tuple: (cropped_image_path, crop_x, crop_y, width, height) if return_crop_coords is True.
        """
        
        # Generate the output file path
        original_filename = os.path.basename(self.reference_image)
        name, ext = os.path.splitext(original_filename)
        
        # Handle special case for ".ome.tiff" files
        if ext == ".tif" and name.endswith(".ome"):
            name = name[:-4]  # Remove ".ome" from the name
            ext = ".ome.tif"
        elif ext == ".tiff" and name.endswith(".ome"):
            name = name[:-4]  # Remove ".ome" from the name
            ext = ".ome.tiff"
        else:
            ext = ".ome.tiff"
        
        output_filename = f"{name}_crop{ext}"
        output_path = os.path.join(self.output_dir, output_filename)
        
        image = pyvips.Image.new_from_file(self.reference_image)

        # Crop the image and get the cropping region
        cropped_image, crop_x, crop_y, width, height = self.crop_image(image)

        # Save the cropped image as OME-TIFF
        slide_io.save_ome_tiff(cropped_image, output_path, tile_wh=1024, compression="jp2k", Q=50, pyramid=True)

        # Read and adjust the annotation coordinates
        if self.annotations_geojson_path:
            annotations = self.read_annotations_coords()
            
            # If the image is .mrxs, we need to consider the bounding box
            bounds_x, bounds_y = 0, 0
            if self.reference_image.endswith('.mrxs'):
                bounds_x = image.get('openslide.bounds-x')
                bounds_y = image.get('openslide.bounds-y')
            
            adjusted_annotations = self.adjust_annotation_coords(annotations, crop_x, crop_y, bounds_x, bounds_y)
            
            # Save the adjusted annotations to a new GeoJSON file
            self.save_annotations_geojson(adjusted_annotations, crop_x, crop_y, width, height)

        if return_crop_coords:
            return output_path, crop_x, crop_y, width, height