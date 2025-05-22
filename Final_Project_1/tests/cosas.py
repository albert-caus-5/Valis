import pyvips
import openslide

# Cargar la imagen MRXS
imagen = pyvips.Image.new_from_file("C:/Users/marqu/OneDrive/Documentos/TFM/base de datos validation/uni_dataset/MRXS/HE/VC22B005113A001010.mrxs")

# Obtener el número de niveles
num_niveles = imagen.get('openslide.level-count')
print(f"Número de niveles: {num_niveles}")

# Imprimir las dimensiones de cada nivel
for i in range(int(num_niveles)):
    nivel_ancho = imagen.get(f'openslide.level[{i}].width')
    nivel_alto = imagen.get(f'openslide.level[{i}].height')
    print(f"Nivel {i}: {nivel_ancho} x {nivel_alto}")


# Obtener las propiedades de la bounding box
bounds_x = imagen.get('openslide.bounds-x')
bounds_y = imagen.get('openslide.bounds-y')
bounds_width = imagen.get('openslide.bounds-width')
bounds_height = imagen.get('openslide.bounds-height')

print(f"Bounding box: x={bounds_x}, y={bounds_y}, width={bounds_width}, height={bounds_height}")