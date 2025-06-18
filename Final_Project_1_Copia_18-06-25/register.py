""" Registration of whole slide images (WSI)

This example shows how to register, warp, and save a collection
of whole slide images (WSI) using the default parameters.

The results directory contains several folders:

1. *data* contains 2 files:
    * a summary spreadsheet of the alignment results, such
    as the registration error between each pair of slides, their
    dimensions, physical units, etc...

    * a pickled version of the registrar. This can be reloaded
    (unpickled) and used later. For example, one could perform
    the registration locally, but then use the pickled object
    to warp and save the slides on an HPC. Or, one could perform
    the registration and use the registrar later to warp
    points in the slide.

2. *overlaps* contains thumbnails showing the how the images
    would look if stacked without being registered, how they
    look after rigid registration, and how they would look
    after non-rigid registration.

3. *rigid_registration* shows thumbnails of how each image
    looks after performing rigid registration.

4. *non_rigid_registration* shows thumbnaials of how each
    image looks after non-rigid registration.

5. *deformation_fields* contains images showing what the
    non-rigid deformation would do to a triangular mesh.
    These can be used to get a better sense of how the
    images were altered by non-rigid warping

6. *processed* shows thumnails of the processed images.
    This are thumbnails of the images that are actually
    used to perform the registration. The pre-processing
    and normalization methods should try to make these
    images look as similar as possible.


After registraation is complete, one should view the
results to determine if they aare acceptable. If they
are, then one can warp and save all of the slides.

"""

import time
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__))) # Calling my_valis instead of installed valis
from my_valis import registration
from my_valis.micro_rigid_registrar import MicroRigidRegistrar # For high resolution rigid registration

class Register:
    def __init__(self, image_paths, save_results_dir, registration_type, reference_img, img_etiquetes_he):
        self.image_paths = image_paths
        self.save_results_dir = save_results_dir
        self.registration_type = registration_type
        print(f"Print de la variable que llegeix de la pantalla els paths{self.image_paths}")
        #Codi meu
        self.reference_img = reference_img
        self.img_etiquetes_he = img_etiquetes_he

    def registration(self, transfer = None, source_image_path=None, target_image_paths=None):
        # Rigid and Non-Rigid registration
        print(f"Registrando imágenes con calidad normal, guardando resultados en {self.save_results_dir}")

        # Create a Valis object and use it to register the slides in slide_src_dir
        start = time.time()
        #if self.reference_img == 1

        registrar = registration.Valis(src_paths = self.image_paths, dst_dir = self.save_results_dir, reference_img_f = self.reference_img, img_etiquetes_he = self.img_etiquetes_he)
        #No fiquem align_to_ref a true pq sinó no agafa la imatge de ref., només serveix pq totes les imatges es registrin amb la de referència i no de forma seqüencial, per nosaltres ens és igual ja que sempre tindrem 2 imatges.
        rigid_registrar, non_rigid_registrar, error_df, iou_rigid, corr_rigid, iou_non_rigid, corr_non_rigid = registrar.register()
        stop = time.time()
        elapsed = stop - start
        print(f"regisration time is {elapsed/60} minutes")

        # If annotation transfer is required, return the image slides
        if transfer != None:
            source_image_slide = registrar.get_slide(source_image_path)
            target_image_slides = []
            for target_image_path in target_image_paths:
                target_image_slide = registrar.get_slide(target_image_path)
                target_image_slides.append(target_image_slide)
            # Shutdown the JVM
            registration.kill_jvm()
            return source_image_slide, target_image_slides
        
        # Shutdown the JVM
        registration.kill_jvm()

        return iou_rigid, corr_rigid, iou_non_rigid, corr_non_rigid
        

    def registration_hd(self, transfer = None, source_image_path=None, target_image_paths=None):
        # Micro-Rigid registration
        print(f"Registrando imágenes con calidad HD, guardando resultados en {self.save_results_dir}")
        
        micro_reg_fraction = 0.25 # Fraction full resolution used for non-rigid registration

        # Perform high resolution rigid registration using the MicroRigidRegistrar
        start = time.time()
        registrar = registration.Valis(self.image_paths, self.save_results_dir, micro_rigid_registrar_cls=MicroRigidRegistrar)
        rigid_registrar, non_rigid_registrar, error_df = registrar.register()

        # Calculate what `max_non_rigid_registration_dim_px` needs to be to do non-rigid registration on an image that is 25% full resolution.
        img_dims = np.array([slide_obj.slide_dimensions_wh[0] for slide_obj in registrar.slide_dict.values()])
        min_max_size = np.min([np.max(d) for d in img_dims])
        img_areas = [np.multiply(*d) for d in img_dims]
        max_img_w, max_img_h = tuple(img_dims[np.argmax(img_areas)])
        micro_reg_size = np.floor(min_max_size*micro_reg_fraction).astype(int)

        # Perform high resolution non-rigid registration
        micro_reg, micro_error = registrar.register_micro(max_non_rigid_registration_dim_px=micro_reg_size)

        stop = time.time()
        elapsed = stop - start
        print(f"regisration time is {elapsed/60} minutes")

        # We can also plot the high resolution matches using `Valis.draw_matches`:
        matches_dst_dir = os.path.join(registrar.dst_dir, "hi_rez_matches")
        registrar.draw_matches(matches_dst_dir)

        # If annotation transfer is required, return the image slides
        if transfer != None:
            source_image_slide = registrar.get_slide(source_image_path)
            target_image_slides = []
            for target_image_path in target_image_paths:
                target_image_slide = registrar.get_slide(target_image_path)
                target_image_slides.append(target_image_slide)
            # Shutdown the JVM
            registration.kill_jvm()
            return source_image_slide, target_image_slides
        
        # Shutdown the JVM
        registration.kill_jvm()
        
