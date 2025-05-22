import argparse
from register import Register
from image_cropper import ImageCropper
from transfering_annotations import AnnotationTransferer

''' Use Examples:
python interfaceinTerminal.py register n_images normal /path/to/image1  /path/to/image2 /path/to/save

python interfaceinTerminal.py transfer n_images  normal /path/to/image1 /path/to/image2 /path/to/save /path/to/source_annotations.geojson /path/to/save_annotations

python interfaceinTerminal.py crop /path/to/image.ome.tiff /path/to/save /path/to/region.geojson --annotations_geojson_path /path/to/annotations.geojson

python interfaceinTerminal.py crop_and_transfer n_images normal /path/to/source_image /path/to/target_image1 ... /path/to/save /path/to/source_annotations.geojson /path/to/save_annotations /path/to/source_roi.geojson /path/to/target_roi1.geojson ...
'''

def register_images(args):
    # Procesar la registración de imágenes
    registration_type = args.registration_type
    image_paths = args.image_paths
    save_dir = args.save_dir
    ref_img = args.ref_image

    if len(image_paths) != args.n_images:
        print(f"Error: Se esperaban {args.n_images} paths de imágenes, pero se recibieron {len(image_paths)}.")
        return

    print(f"Registrando {args.n_images} imágenes con calidad {registration_type}, guardando resultados en {save_dir}")

    register = Register(image_paths, save_dir, registration_type, ref_img)
        
    if registration_type == "normal":
        register.registration()
    elif registration_type == "hd":
        register.registration_hd()
    else:
        print("¡Tipo de registro inválido seleccionado!")


def transfer_annotations(args):
    # Procesar la transferencia de anotaciones
    registration_type = args.registration_type
    image_paths = args.image_paths
    save_dir = args.save_dir
    geojson_path = args.geojson_path
    save_geojson_path = args.save_geojson_path

    if len(image_paths) != args.n_images:
        print(f"Error: Se esperaban {args.n_images} paths de imágenes, pero se recibieron {len(image_paths)}.")
        return

    source_image_path = image_paths[0]
    target_image_paths = image_paths[1:]

    print(f"Transfiriendo anotaciones de {source_image_path} a {target_image_paths} con calidad {registration_type}, guardando resultados en {save_dir}")

    transferer = AnnotationTransferer(
        save_dir,
        source_image_path,
        target_image_paths,
        geojson_path,
        save_geojson_path,
        registration_type
    )
    
    transferer.transfer_annotations()

def crop_image(args):
    # Procesar el recorte de la imagen
    image_path = args.image_path
    save_image_path = args.save_dir
    geojson_path = args.geojson_path
    annotations_geojson_path = args.annotations_geojson_path if args.annotations_geojson_path else None

    print(f"Recortando {image_path} con la región de {geojson_path}, guardando en {save_image_path}")
    
    if annotations_geojson_path:
        print(f"Incluyendo anotaciones de {annotations_geojson_path}")
    
    cropper = ImageCropper(image_path, save_image_path, geojson_path, annotations_geojson_path)
    cropper.crop_and_save_image()

def crop_and_transfer(args):
    # Procesar el recorte y la transferencia de anotaciones
    registration_type = args.registration_type
    image_paths = args.image_paths
    save_dir = args.save_dir
    geojson_path = args.geojson_path
    save_geojson_path = args.save_geojson_path
    geojson_source_roi_path = args.geojson_source_roi_path
    geojson_target_roi_paths = args.geojson_target_roi_paths

    if len(image_paths) != args.n_images:
        print(f"Error: Se esperaban {args.n_images} paths de imágenes, pero se recibieron {len(image_paths)}.")
        return

    source_image_path = image_paths[0]
    target_image_paths = image_paths[1:]

    print(f"Recortando y transfiriendo anotaciones de {source_image_path} a {target_image_paths} con calidad {registration_type}, guardando resultados en {save_dir}")

    transferer = AnnotationTransferer(
        save_dir,
        source_image_path,
        target_image_paths,
        geojson_path,
        save_geojson_path,
        registration_type
    )
    transferer.cropNtransfer_annotations(geojson_source_roi_path, geojson_target_roi_paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cell Image Processing CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subparser para 'register'
    parser_register = subparsers.add_parser('register', help="Register multiple images")
    parser_register.add_argument('n_images', type=int, help="Number of images to register")
    parser_register.add_argument('registration_type', choices=['normal', 'hd'], help="Type of registration (normal or hd)")
    parser_register.add_argument('image_paths', nargs='+', help="Paths to the images to be registered")
    parser_register.add_argument('ref_image', help="Indicar el nom de la imatge de referència (ha de ser una de les imatges afegides anteriorment)")
    parser_register.add_argument('save_dir', help="Directory to save the registration results")
    parser_register.set_defaults(func=register_images)

    # Subparser para 'transfer'
    parser_transfer = subparsers.add_parser('transfer', help="Transfer annotations from source to target images")
    parser_transfer.add_argument('n_images', type=int, help="Number of images involved in the transfer (source + targets)")
    parser_transfer.add_argument('registration_type', choices=['normal', 'hd'], help="Type of registration (normal or hd)")
    parser_transfer.add_argument('image_paths', nargs='+', help="Paths to the source and target images (first is source, rest are targets)")
    parser_transfer.add_argument('save_dir', help="Directory to save the transfer results")
    parser_transfer.add_argument('geojson_path', help="Path to the source GeoJSON file with annotations")
    parser_transfer.add_argument('save_geojson_path', help="Directory to save the new GeoJSON annotations")
    parser_transfer.set_defaults(func=transfer_annotations)

    # Subparser para 'crop'
    parser_crop = subparsers.add_parser('crop', help="Crop an image using a GeoJSON region")
    parser_crop.add_argument('image_path', help="Path to the image to be cropped")
    parser_crop.add_argument('save_dir', help="Directory to save the cropped image")
    parser_crop.add_argument('geojson_path', help="Path to the GeoJSON file defining the cropping region")
    parser_crop.add_argument('--annotations_geojson_path', help="Optional: Path to a GeoJSON file with annotations in the crop region", default=None)
    parser_crop.set_defaults(func=crop_image)

    # Subparser para 'crop_and_transfer'
    parser_crop_and_transfer = subparsers.add_parser('crop_and_transfer', help="Crop source image and transfer annotations")
    parser_crop_and_transfer.add_argument('n_images', type=int, help="Number of images involved in the process (source + targets)")
    parser_crop_and_transfer.add_argument('registration_type', choices=['normal', 'hd'], help="Type of registration (normal or hd)")
    parser_crop_and_transfer.add_argument('image_paths', nargs='+', help="Paths to the source and target images (first is source, rest are targets)")
    parser_crop_and_transfer.add_argument('save_dir', help="Directory to save the results")
    parser_crop_and_transfer.add_argument('geojson_path', help="Path to the source GeoJSON file with annotations")
    parser_crop_and_transfer.add_argument('save_geojson_path', help="Directory to save the new GeoJSON annotations")
    parser_crop_and_transfer.add_argument('geojson_source_roi_path', help="Path to the GeoJSON file defining the cropping region for the source image")
    parser_crop_and_transfer.add_argument('geojson_target_roi_paths', nargs='+', help="Paths to the GeoJSON files defining the cropping regions for the target images")
    parser_crop_and_transfer.set_defaults(func=crop_and_transfer)

    # Parsear los argumentos y llamar a la función adecuada
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
    else:
        args.func(args)