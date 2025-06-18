import pyvips
import json
from valis import slide_io
from PIL import Image

geojson_path = "C:/Users/marqu/OneDrive/Documentos/TFM/base de datos validation/results/results prueba/5113.geojson"

imagen = pyvips.Image.new_from_file("C:/Users/marqu/OneDrive/Documentos/TFM/base de datos validation/uni_dataset/MRXS/HE/VC22B005113A001010.mrxs")

'''x1 = 0
y1 = 100842
x2 = 120196
y2 = 158224
'''

with open(geojson_path, 'r') as f:
    data = json.load(f)

# Extract the coordinates of the first polygon
features = data['features']
if not features:
    raise ValueError("No features found in the GeoJSON.")

geometry = features[0]['geometry']
if geometry['type'] != 'Polygon':
    raise ValueError("The geometry is not a polygon.")

coords = geometry['coordinates'][0]

# Obtener las propiedades de la bounding box
bounds_x = imagen.get('openslide.bounds-x')
bounds_y = imagen.get('openslide.bounds-y')
bounds_width = imagen.get('openslide.bounds-width')
bounds_height = imagen.get('openslide.bounds-height')

# Ajustar las coordenadas utilizando el offset de la bounding box
coords_ajustadas = []
for coord in coords:
    x_ajustado = coord[0] + int(bounds_x)
    y_ajustado = coord[1] + int(bounds_y)
    coords_ajustadas.append([x_ajustado, y_ajustado])

# Obtener las coordenadas ajustadas del GeoJSON
x1, y1 = coords_ajustadas[0]
x2, y2 = coords_ajustadas[2]
ancho_recorte = x2 - x1
alto_recorte = y2 - y1

recorte = imagen.crop(x1, y1, ancho_recorte, alto_recorte)

path = "C:/Users/marqu/OneDrive/Documentos/TFM/base de datos validation/results/results prueba/cropped"
#recorte.save("recorte.mrxs")
slide_io.save_ome_tiff(recorte, path, tile_wh=1024, compression="lzw", Q=100, pyramid=True)



# Redimensionar la imagen recortada para que tenga dimensiones más pequeñas
#recorte_redimensionado = recorte.resize(0.01)  # Por ejemplo, reduce el tamaño a la mitad

# Guardar el recorte en formato JPEG con baja calidad
#recorte_redimensionado.write_to_file("C:/Users/marqu/OneDrive/Documentos/TFM/base de datos validation/results/results prueba/imagen_recortada.jpg", Q=10)

