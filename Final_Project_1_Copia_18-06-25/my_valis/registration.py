"""
Classes and functions to register a collection of images
"""

import time as pytime
import traceback
import re
import os
import numpy as np
import pathlib
from skimage import transform, exposure, filters
import numpy as np
from skimage.io import imread
from skimage.util import img_as_float
from skimage.transform import resize
from skimage import img_as_ubyte
from skimage.exposure import rescale_intensity
import imageio.v3 as iio
from time import time
import tqdm
import pandas as pd
import pickle
import colour
import pyvips
import cv2
from scipy import ndimage
import shapely
from copy import deepcopy
from pprint import pformat
import json
from colorama import Fore
from itertools import chain
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import matplotlib.pyplot as plt
import openslide
from pathlib import Path

from . import feature_matcher
from . import serial_rigid
from . import feature_detectors
from . import non_rigid_registrars
from . import valtils
from . import preprocessing
from . import slide_tools
from . import slide_io
from . import viz
from . import warp_tools
from . import serial_non_rigid

pyvips.cache_set_max(0)

# Destination directories #
CONVERTED_IMG_DIR = "images"
PROCESSED_IMG_DIR = "processed"
RIGID_REG_IMG_DIR = "rigid_registration"
NON_RIGID_REG_IMG_DIR = "non_rigid_registration"
DEFORMATION_FIELD_IMG_DIR = "deformation_fields"
OVERLAP_IMG_DIR = "overlaps"
REG_RESULTS_DATA_DIR = "data"
MICRO_REG_DIR = "micro_registration"

#Codi canviat
DISPLACEMENT_DIRS = "displacements"
#

MASK_DIR = "masks"

# Default image processing #
DEFAULT_BRIGHTFIELD_CLASS = preprocessing.ColorfulStandardizer
DEFAULT_BRIGHTFIELD_PROCESSING_ARGS = {'c': preprocessing.DEFAULT_COLOR_STD_C, "h": 0}
DEFAULT_FLOURESCENCE_CLASS = preprocessing.ChannelGetter
DEFAULT_FLOURESCENCE_PROCESSING_ARGS = {"channel": "dapi", "adaptive_eq": True}
DEFAULT_NORM_METHOD = "img_stats"

# Default rigid registration parameters #
DEFAULT_FD = feature_detectors.VggFD
DEFAULT_TRANSFORM_CLASS = transform.SimilarityTransform
DEFAULT_MATCH_FILTER = feature_matcher.Matcher(match_filter_method=feature_matcher.RANSAC_NAME)
DEFAULT_SIMILARITY_METRIC = "n_matches"
DEFAULT_AFFINE_OPTIMIZER_CLASS = None

#Codi canviat
DEFAULT_MAX_PROCESSED_IMG_SIZE = 1850 #850
DEFAULT_MAX_IMG_DIM = 1850 # 850
DEFAULT_THUMBNAIL_SIZE = 1850 #500
#

DEFAULT_MAX_NON_RIGID_REG_SIZE = 3000

# Tiled non-rigid registration arguments
TILER_THRESH_GB = 10
DEFAULT_NR_TILE_WH = 512

# Rigid registration kwarg keys #
AFFINE_OPTIMIZER_KEY = "affine_optimizer"
TRANSFORMER_KEY = "transformer"
SIM_METRIC_KEY = "similarity_metric"
FD_KEY = "feature_detector"
MATCHER_KEY = "matcher"
NAME_KEY = "name"
IMAGES_ORDERD_KEY = "imgs_ordered"
REF_IMG_KEY = "reference_img_f"
QT_EMMITER_KEY = "qt_emitter"
TFORM_SRC_SHAPE_KEY = "transformation_src_shape_rc"
TFORM_DST_SHAPE_KEY = "transformation_dst_shape_rc"
TFORM_MAT_KEY = "M"
CHECK_REFLECT_KEY = "check_for_reflections"

# Rigid registration kwarg keys #
NON_RIGID_REG_CLASS_KEY = "non_rigid_reg_class"
NON_RIGID_REG_PARAMS_KEY = "non_rigid_reg_params"
NON_RIGID_USE_XY_KEY = "moving_to_fixed_xy"
NON_RIGID_COMPOSE_KEY = "compose_transforms"

# Default non-rigid registration parameters #
DEFAULT_NON_RIGID_CLASS = non_rigid_registrars.OpticalFlowWarper
DEFAULT_NON_RIGID_KWARGS = {}

# Cropping options
CROP_OVERLAP = "overlap"
CROP_REF = "reference"
CROP_NONE = "all"

# Messages
WARP_ANNO_MSG = "Warping annotations"
CONVERT_MSG = "Converting images"
DENOISE_MSG = "Denoising images"
PROCESS_IMG_MSG = "Processing images"
NORM_IMG_MSG = "Normalizing images"
TRANSFORM_MSG = "Finding rigid transforms"
PREP_NON_RIGID_MSG = "Preparing images for non-rigid registration"
MEASURE_MSG = "Measuring error"
SAVING_IMG_MSG = "Saving images"

PROCESS_IMG_MSG, NORM_IMG_MSG, DENOISE_MSG = valtils.pad_strings([PROCESS_IMG_MSG, NORM_IMG_MSG, DENOISE_MSG])


def init_jvm(jar=None, mem_gb=10):
    """Initialize JVM for BioFormats
    """
    slide_io.init_jvm(jar=None, mem_gb=10)


def kill_jvm():
    """Kill JVM for BioFormats
    """
    slide_io.kill_jvm()


def load_registrar(src_f):
    """Load a Valis object

    Parameters
    ----------
    src_f : string
        Path to pickled Valis object

    Returns
    -------
    registrar : Valis

        Valis object used for registration

    """
    registrar = pickle.load(open(src_f, 'rb'))

    data_dir = registrar.data_dir
    read_data_dir = os.path.split(src_f)[0]

    # If registrar has moved, will need to update paths to results
    # and displacement fields
    if data_dir != read_data_dir:
        new_dst_dir = os.path.split(read_data_dir)[0]
        registrar.dst_dir = new_dst_dir
        registrar.set_dst_paths()

        for slide_obj in registrar.slide_dict.values():
            slide_obj.update_results_img_paths()

    return registrar


class Slide(object):
    """Stores registration info and warps slides/points

    `Slide` is a class that stores registration parameters
    and other metadata about a slide. Once registration has been
    completed, `Slide` is also able warp the slide and/or points
    using the same registration parameters. Warped slides can be saved
    as ome.tiff images with valid ome-xml.

    Attributes
    ----------
    src_f : str
        Path to slide.

    image: ndarray
        Image to registered. Taken from a level in the image pyramid.
        However, image may be resized to fit within the `max_image_dim_px`
        argument specified when creating a `Valis` object.

    val_obj : Valis
        The "parent" object that registers all of the slide.

    reader : SlideReader
        Object that can read slides and collect metadata.

    original_xml : str
        Xml string created by bio-formats

    img_type : str
        Whether the image is "brightfield" or "fluorescence"

    is_rgb : bool
        Whether or not the slide is RGB.

    slide_shape_rc : tuple of int
        Dimensions of the largest resolution in the slide, in the form
        of (row, col).

    series : int
        Slide series to be read

    slide_dimensions_wh : ndarray
        Dimensions of all images in the pyramid (width, height).

    resolution : float
        Physical size of each pixel.

    units : str
        Physical unit of each pixel.

    name : str
        Name of the image. Usually `img_f` but with the extension removed.

    processed_img : ndarray
        Image used to perform registration

    rigid_reg_mask : ndarray
        Mask of convex hulls covering tissue in unregistered image.
        Could be used to mask `processed_img` before rigid registration

    non_rigid_reg_mask : ndarray
        Created by combining rigidly warped `rigid_reg_mask` in all
        other slides.

    stack_idx : int
        Position of image in sorted Z-stack

    processed_img_f : str
        Path to thumbnail of the processed `image`.

    rigid_reg_img_f : str
        Path to thumbnail of rigidly aligned `image`.

    non_rigid_reg_img_f : str
        Path to thumbnail of non-rigidly aligned `image`.

    processed_img_shape_rc : tuple of int
        Shape (row, col) of the processed image used to find the
        transformation parameters. Maximum dimension will be less or
        equal to the `max_processed_image_dim_px` specified when
        creating a `Valis` object. As such, this may be smaller than
        the image's shape.

    aligned_slide_shape_rc : tuple of int
        Shape (row, col) of aligned slide, based on the dimensions in the 0th
        level of they pyramid. In

    reg_img_shape_rc : tuple of int
        Shape (row, col) of the registered image

    M : ndarray
        Rigid transformation matrix that aligns `image` to the previous
        image in the stack. Found using the processed copy of `image`.

    bk_dxdy : ndarray
        (2, N, M) numpy array of pixel displacements in
        the x and y directions. dx = bk_dxdy[0], and dy=bk_dxdy[1]. Used
        to warp images. Found using the rigidly aligned version of the
        processed image.

    fwd_dxdy : ndarray
        Inverse of `bk_dxdy`. Used to warp points.

    _bk_dxdy_f : str
        Path to file containing bk_dxdy, if saved

    _fwd_dxdy_f : str
        Path to file containing fwd_dxdy, if saved

    _bk_dxdy_np : ndarray
        `bk_dxdy` as a numpy array. Only not None if `bk_dxdy` becomes
        associated with a file

    _fwd_dxdy_np : ndarray
        `fwd_dxdy` as a numpy array. Only not None if `fwd_dxdy` becomes
        associated with a file

    stored_dxdy : bool
        Whether or not the non-rigid displacements are saved in a file
        Should only occur if image is very large.

    fixed_slide : Slide
        Slide object to which this one was aligned.

    xy_matched_to_prev : ndarray
        Coordinates (x, y) of features in `image` that had matches in the
        previous image. Will have shape (N, 2)

    xy_in_prev : ndarray
        Coordinates (x, y) of features in the previous that had matches
        to those in `image`. Will have shape (N, 2)

    xy_matched_to_prev_in_bbox : ndarray
        Subset of `xy_matched_to_prev` that were within `overlap_mask_bbox_xywh`.
        Will either have shape (N, 2) or (M, 2), with M < N.

    xy_in_prev_in_bbox : ndarray
        Subset of `xy_in_prev` that were within `overlap_mask_bbox_xywh`.
        Will either have shape (N, 2) or (M, 2), with M < N.

    crop : str
        Crop method

    bg_px_pos_rc : tuple
        Position of pixel that has the background color

    bg_color : list, optional
        Color of background pixels

    is_empty : bool
        True if the image is empty (i.e. contains only 1 value)

    """

    def __init__(self, src_f, image, val_obj, reader, name=None):
        """
        Parameters
        ----------
        src_f : str
            Path to slide.

        image: ndarray
            Image to registered. Taken from a level in the image pyramid.
            However, image may be resized to fit within the `max_image_dim_px`
            argument specified when creating a `Valis` object.

        val_obj : Valis
            The "parent" object that registers all of the slide.

        reader : SlideReader
            Object that can read slides and collect metadata.

        name : str, optional
            Name of slide. If None, it will be `src_f` with the extension removed

        """

        self.src_f = src_f
        self.image = image
        self.val_obj = val_obj
        self.reader = reader

        # Metadata #
        self.is_rgb = reader.metadata.is_rgb
        self.img_type = reader.guess_image_type()
        self.slide_shape_rc = reader.metadata.slide_dimensions[0][::-1]
        self.series = reader.series
        self.slide_dimensions_wh = reader.metadata.slide_dimensions
        self.resolution = np.mean(reader.metadata.pixel_physical_size_xyu[0:2])
        self.units = reader.metadata.pixel_physical_size_xyu[2]
        self.original_xml = reader.metadata.original_xml

        if name is None:
            name = valtils.get_name(src_f)

        self.name = name

        # To be filled in during registration #
        self.processed_img = None
        self.rigid_reg_mask = None
        self.non_rigid_reg_mask = None
        self.stack_idx = None

        self.aligned_slide_shape_rc = None
        self.processed_img_shape_rc = None
        self.reg_img_shape_rc = None
        self.M = None
        self.bk_dxdy = None
        self.fwd_dxdy = None

        self.stored_dxdy = False
        self._bk_dxdy_f = None
        self._fwd_dxdy_f = None
        self._bk_dxdy_np = None
        self._fwd_dxdy_np = None
        self.processed_img_f = None
        self.rigid_reg_img_f = None
        self.non_rigid_reg_img_f = None

        self.fixed_slide = None
        self.xy_matched_to_prev = None
        self.xy_in_prev = None
        self.xy_matched_to_prev_in_bbox = None
        self.xy_in_prev_in_bbox = None

        self.crop = None
        self.bg_px_pos_rc = (0, 0)
        self.bg_color = None

        self.is_empty = self.check_if_empty(image)

    def check_if_empty(self, img):
        """Check if the image is empty

        Return
        ------
        is_empty : bool
            Whether or not the image is empty

        """

        is_empty = img.min() == img.max()

        return is_empty

    def slide2image(self, level, series=None, xywh=None):
        """Convert slide to image

        Parameters
        -----------
        level : int
            Pyramid level

        series : int, optional
            Series number. Defaults to 0

        xywh : tuple of int, optional
            The region to be sliced from the slide. If None,
            then the entire slide will be converted. Otherwise
            xywh is the (top left x, top left y, width, height) of
            the region to be sliced.

        Returns
        -------
        img : ndarray
            An image of the slide or the region defined by xywh

        """

        img = self.reader.slide2image(level=level, series=series, xywh=xywh)

        return img

    def slide2vips(self, level, series=None, xywh=None):
        """Convert slide to pyvips.Image

        Parameters
        -----------
        level : int
            Pyramid level

        series : int, optional
            Series number. Defaults to 0

        xywh : tuple of int, optional
            The region to be sliced from the slide. If None,
            then the entire slide will be converted. Otherwise
            xywh is the (top left x, top left y, width, height) of
            the region to be sliced.

        Returns
        -------
        vips_slide : pyvips.Image
            An of the slide or the region defined by xywh

        """

        vips_img, res = self.reader.slide2vips(level=level, series=series, xywh=xywh)

        return vips_img

    def get_aligned_to_ref_slide_crop_xywh(self, ref_img_shape_rc, ref_M, scaled_ref_img_shape_rc=None):
        """Get bounding box used to crop slide to fit in reference image

        Parameters
        ----------
        ref_img_shape_rc : tuple of int
            shape of reference image used to find registration parameters, i.e. processed image)

        ref_M : ndarray
            Transformation matrix for the reference image

        scaled_ref_img_shape_rc : tuple of int, optional
            shape of scaled image with shape `img_shape_rc`, i.e. slide corresponding
            to the image used to find the registration parameters.

        Returns
        -------
        crop_xywh : tuple of int
            Bounding box of crop area (XYWH)

        mask : ndarray
            Mask covering reference image

        """

        mask , _ = self.val_obj.get_crop_mask(CROP_REF)

        if scaled_ref_img_shape_rc is not None:
            sxy = np.array([*scaled_ref_img_shape_rc[::-1]]) / np.array([*ref_img_shape_rc[::-1]])
        else:
            scaled_ref_img_shape_rc = ref_img_shape_rc
            sxy = np.ones(2)

        reg_txy = -ref_M[0:2, 2]
        slide_xywh = (*reg_txy*sxy, *scaled_ref_img_shape_rc[::-1])

        return slide_xywh, mask

    def get_overlap_crop_xywh(self, warped_img_shape_rc, scaled_warped_img_shape_rc=None):
        """Get bounding box used to crop slide to where all slides overlap

        Parameters
        ----------
        warped_img_shape_rc : tuple of int
            shape of registered image

        warped_scaled_img_shape_rc : tuple of int, optional
            shape of scaled registered image (i.e. registered slied)

        Returns
        -------
        crop_xywh : tuple of int
            Bounding box of crop area (XYWH)

        """
        mask , mask_bbox_xywh = self.val_obj.get_crop_mask(CROP_OVERLAP)

        if scaled_warped_img_shape_rc is not None:
            sxy = np.array([*scaled_warped_img_shape_rc[::-1]]) / np.array([*warped_img_shape_rc[::-1]])
        else:
            sxy = np.ones(2)

        to_slide_transformer = transform.SimilarityTransform(scale=sxy)
        overlap_bbox = warp_tools.bbox2xy(mask_bbox_xywh)
        scaled_overlap_bbox = to_slide_transformer(overlap_bbox)
        scaled_overlap_xywh = warp_tools.xy2bbox(scaled_overlap_bbox)

        scaled_overlap_xywh[2:] = np.ceil(scaled_overlap_xywh[2:])
        scaled_overlap_xywh = tuple(scaled_overlap_xywh.astype(int))

        return scaled_overlap_xywh, mask

    def get_crop_xywh(self, crop, out_shape_rc=None):
        """Get bounding box used to crop aligned slide

        Parameters
        ----------

        out_shape_rc : tuple of int, optional
            If crop is "reference", this should be the shape of scaled reference image, such
            as the unwarped slide that corresponds to the unwarped processed reference image.

            If crop is "overlap", this should be the shape of the registered slides.


        Returns
        -------
        crop_xywh : tuple of int
            Bounding box of crop area (XYWH)

        mask : ndarray
            Mask, before crop
        """

        ref_slide = self.val_obj.get_ref_slide()
        if crop == CROP_REF:
            transformation_shape_rc = np.array(ref_slide.processed_img_shape_rc)
            crop_xywh, mask = self.get_aligned_to_ref_slide_crop_xywh(ref_img_shape_rc=transformation_shape_rc,
                                                                      ref_M=ref_slide.M,
                                                                      scaled_ref_img_shape_rc=out_shape_rc)
        elif crop == CROP_OVERLAP:
            transformation_shape_rc = np.array(ref_slide.reg_img_shape_rc)
            crop_xywh, mask = self.get_overlap_crop_xywh(warped_img_shape_rc=transformation_shape_rc,
                                                         scaled_warped_img_shape_rc=out_shape_rc)

        return crop_xywh, mask

    def get_crop_method(self, crop):
        """Get string or logic defining how to crop the image
        """
        if crop is True:
            crop_method = self.crop
        else:
            crop_method = crop

        do_crop = crop_method in [CROP_REF, CROP_OVERLAP]

        if do_crop:
            return crop_method
        else:
            return False

    def get_bg_color_px_pos(self):
        """Get position of pixel that has color used for background
        """
        if self.img_type == slide_tools.IHC_NAME:
            # RGB. Get brightest pixel
            eps = np.finfo("float").eps
            with colour.utilities.suppress_warnings(colour_usage_warnings=True):
                if 1 < self.image.max() <= 255 and np.issubdtype(self.image.dtype, np.integer):
                    cam = colour.convert(self.image/255 + eps, 'sRGB', 'CAM16UCS')
                else:
                    cam = colour.convert(self.image + eps, 'sRGB', 'CAM16UCS')

            lum = cam[..., 0]
            bg_px = np.unravel_index(np.argmax(lum, axis=None), lum.shape)
        else:
            # IF. Get darkest pixel
            sum_img = self.image.sum(axis=2)
            bg_px = np.unravel_index(np.argmin(sum_img, axis=None), sum_img.shape)

        self.bg_px_pos_rc = bg_px
        self.bg_color = list(self.image[bg_px])

    def update_results_img_paths(self):
        n_digits = len(str(self.val_obj.size))
        stack_id = str.zfill(str(self.stack_idx), n_digits)

        self.processed_img_f = os.path.join(self.val_obj.processed_dir, self.name + ".png")
        self.rigid_reg_img_f = os.path.join(self.val_obj.reg_dst_dir, f"{stack_id}_f{self.name}.png")
        self.non_rigid_reg_img_f = os.path.join(self.val_obj.non_rigid_dst_dir, f"{stack_id}_f{self.name}.png")
        if self.stored_dxdy:
            bk_dxdy_f, fwd_dxdy_f = self.get_displacement_f()
            self._bk_dxdy_f = bk_dxdy_f
            self._fwd_dxdy_f = fwd_dxdy_f

    def get_displacement_f(self):
        bk_dxdy_f = os.path.join(self.val_obj.displacements_dir, f"{self.name}_bk_dxdy.tiff")
        fwd_dxdy_f = os.path.join(self.val_obj.displacements_dir, f"{self.name}_fwd_dxdy.tiff")

        return bk_dxdy_f, fwd_dxdy_f

    def get_bk_dxdy(self):
        if self.stored_dxdy:
            bk_dxdy_f, _ = self.get_displacement_f()
            cropped_bk_dxdy = pyvips.Image.new_from_file(bk_dxdy_f)
            full_bk_dxdy = self.val_obj.pad_displacement(cropped_bk_dxdy,
                self.val_obj._full_displacement_shape_rc,
                self.val_obj._non_rigid_bbox)

            return full_bk_dxdy
        else:
            return self._bk_dxdy_np

    def set_bk_dxdy(self, bk_dxdy):
        
        #Codi meu
        """Assigna el camp de desplaçament i el desa en el directori displacements_dir si no està buit"""
        
        #print(f"Executant set_bk_dxdy per {self.name}")
    
        # Comprovar si bk_dxdy és None
        if bk_dxdy is None:
            #print(f"⚠️ La matriu bk_dxdy per {self.name} està buida i no es desarà.")
            return
    
        # Comprovar si bk_dxdy és un objecte VIPS i convertir-lo a NumPy
        if isinstance(bk_dxdy, pyvips.Image):
            #print("Converting bk_dxdy from pyvips.Image to NumPy array...")
            bk_dxdy = warp_tools.vips2numpy(bk_dxdy)
    
        # Comprovar si és realment un ndarray
        #

        if not isinstance(bk_dxdy, np.ndarray):
        
        #Codi meu

            #print(f"ERROR: bk_dxdy no és de tipus ndarray per {self.name}, sinó {type(bk_dxdy)}")
            return
    
        # Comprovar si tots els valors són 0
        if np.all(bk_dxdy == 0):
            #print(f"⚠️ La matriu bk_dxdy per {self.name} està buida (tots els valors són zero) i no es desarà.")
            return
    
        # Desa la versió en memòria
        #
        self._bk_dxdy_np = bk_dxdy


        #Codi meu
        # Obtenir el directori de displacements
        displacements_dir = self.val_obj.displacements_dir
        if displacements_dir is None:
            print(f"ERROR: displacements_dir és None per {self.name}")
            return
    
        #print(f"Guardant bk_dxdy a {displacements_dir}")
    
        # Crear el directori si no existeix
        os.makedirs(displacements_dir, exist_ok=True)
    
        # Guardar el fitxer
        #displacement_file = os.path.join(displacements_dir, f"{self.name}_bk_dxdy.npy")
        #np.save(displacement_file, self._bk_dxdy_np)
        #print(f"Desat el camp de moviment a {displacement_file}")
    
        # Desa la ruta del fitxer
        #self._bk_dxdy_f = displacement_file
        #
    
    bk_dxdy = property(fget=get_bk_dxdy,
                       fset=set_bk_dxdy,
                       doc="Get and set backwards displacements")

    def get_fwd_dxdy(self):
        if self.stored_dxdy:
            _, fwd_dxdy_f = self.get_displacement_f()
            cropped_fwd_dxdy = pyvips.Image.new_from_file(fwd_dxdy_f)
            full_fwd_dxdy = self.val_obj.pad_displacement(cropped_fwd_dxdy,
                self.val_obj._full_displacement_shape_rc,
                self.val_obj._non_rigid_bbox)

            return full_fwd_dxdy

        else:
            return self._fwd_dxdy_np
            
            
            
    def set_fwd_dxdy(self, fwd_dxdy):
        
        #Codi meu
        """Assigna el camp de desplaçament invers i el desa en el directori displacements_dir si no està buit"""
        
        #print(f"Executant set_fwd_dxdy per {self.name}")
    
        # Comprovar si fwd_dxdy és None
        if fwd_dxdy is None:
            #print(f"⚠️ La matriu fwd_dxdy per {self.name} està buida i no es desarà.")
            return
    
        # Comprovar si fwd_dxdy és un objecte VIPS i convertir-lo a NumPy
        #

        if isinstance(fwd_dxdy, pyvips.Image):
        
        #Codi meu
            #print("Converting fwd_dxdy from pyvips.Image to NumPy array...")
            fwd_dxdy = warp_tools.vips2numpy(fwd_dxdy)
    
        # Comprovar si és realment un ndarray
        if not isinstance(fwd_dxdy, np.ndarray):
            #print(f"ERROR: fwd_dxdy no és de tipus ndarray per {self.name}, sinó {type(fwd_dxdy)}")
            return
    
        # Comprovar si tots els valors són 0
        if np.all(fwd_dxdy == 0):
            #print(f"⚠️ La matriu fwd_dxdy per {self.name} està buida (tots els valors són zero) i no es desarà.")
            return
    
        # Desa la versió en memòria
        #

        self._fwd_dxdy_np = fwd_dxdy

        #Codi meu
        # Obtenir el directori de displacements
        displacements_dir = self.val_obj.displacements_dir
        if displacements_dir is None:
            print(f"ERROR: displacements_dir és None per {self.name}")
            return
    
        #print(f"Guardant fwd_dxdy a {displacements_dir}")
    
        # Crear el directori si no existeix
        os.makedirs(displacements_dir, exist_ok=True)
    
        # Guardar el fitxer
        #displacement_file = os.path.join(displacements_dir, f"{self.name}_fwd_dxdy.npy")
        #np.save(displacement_file, self._fwd_dxdy_np)
        #print(f"Desat el camp de moviment invers a {displacement_file}")
    
        # Desa la ruta del fitxer
        #self._fwd_dxdy_f = displacement_file
        #


    fwd_dxdy = property(fget=get_fwd_dxdy,
                        fset=set_fwd_dxdy,
                        doc="Get forward displacements")
                        
                        
                        
    #Codi meu                    
    def save_M_field(self):
        """ Guarda la matriu de transformació M al directori displacements/ si no està buida """
        
        displacements_dir = os.path.join(self.val_obj.dst_dir, "displacements")
        os.makedirs(displacements_dir, exist_ok=True)
        
        
        # Guardar la matriu de transformació rígida M
        if self.M is None or np.all(self.M == 0):
            print(f"⚠️ La matriu M per {self.name} està buida i no es desarà.")
        else:
            M_file = os.path.join(displacements_dir, f"{self.name}_M.npy")
            np.save(M_file, self.M)
            print(f"Matriu de transformació M desada a: {M_file}")
        
    #
            

    def warp_img(self, img=None, non_rigid=True, crop=True, interp_method="bicubic"):
        """Warp an image using the registration parameters

        img : ndarray, optional
            The image to be warped. If None, then Slide.image
            will be warped.

        non_rigid : bool
            Whether or not to conduct non-rigid warping. If False,
            then only a rigid transformation will be applied.

        crop: bool, str
            How to crop the registered images. If `True`, then the same crop used
            when initializing the `Valis` object will be used. If `False`, the
            image will not be cropped. If "overlap", the warped slide will be
            cropped to include only areas where all images overlapped.
            "reference" crops to the area that overlaps with the reference image,
            defined by `reference_img_f` when initialzing the `Valis object`.

        interp_method : str
            Interpolation method used when warping slide. Default is "bicubic"

        Returns
        -------
        warped_img : ndarray
            Warped copy of `img`

        """

        if img is None:
            img = self.image

        if non_rigid:
            dxdy = self.bk_dxdy
        else:
            dxdy = None

        if isinstance(img, pyvips.Image):
            img_shape_rc = (img.width, img.height)
            img_dim = img.bands
        else:
            img_shape_rc = img.shape[0:2]
            img_dim = img.ndim

        if not np.all(img_shape_rc == self.processed_img_shape_rc):
            msg = ("scaling transformation for image with different shape. "
                   "However, without knowing all of other image's shapes, "
                   "the scaling may not be the same for all images, and so "
                   "may not overlap."
                   )
            valtils.print_warning(msg)
            same_shape = False
            img_scale_rc = np.array(img_shape_rc)/(np.array(self.processed_img_shape_rc))
            out_shape_rc = self.val_obj.get_aligned_slide_shape(img_scale_rc)


        else:
            same_shape = True
            out_shape_rc = self.reg_img_shape_rc

        if isinstance(crop, bool) or isinstance(crop, str):
            crop_method = self.get_crop_method(crop)
            if crop_method is not False:
                if crop_method == CROP_REF:
                    ref_slide = self.val_obj.get_ref_slide()
                    if not same_shape:
                        scaled_shape_rc = np.array(ref_slide.processed_img_shape_rc)*img_scale_rc
                    else:
                        scaled_shape_rc = ref_slide.processed_img_shape_rc
                elif crop_method == CROP_OVERLAP:
                    scaled_shape_rc = out_shape_rc

                bbox_xywh, _ = self.get_crop_xywh(crop_method, scaled_shape_rc)
            else:
                bbox_xywh = None

        elif isinstance(crop[0], (int, float)) and len(crop) == 4:
                bbox_xywh = crop
        else:
            bbox_xywh = None

        if img_dim == self.image.ndim:
            bg_color = self.bg_color
        else:
            bg_color = None

        warped_img = \
            warp_tools.warp_img(img, M=self.M,
                                bk_dxdy=dxdy,
                                out_shape_rc=out_shape_rc,
                                transformation_src_shape_rc=self.processed_img_shape_rc,
                                transformation_dst_shape_rc=self.reg_img_shape_rc,
                                bbox_xywh=bbox_xywh,
                                bg_color=bg_color,
                                interp_method=interp_method)

        return warped_img

    def warp_img_from_to(self, img, to_slide_obj,
                         dst_slide_level=0, non_rigid=True, interp_method="bicubic", bg_color=None):

        """Warp an image from this slide onto another unwarped slide

        Note that if `img` is a labeled image then it is recommended to set `interp_method` to "nearest"

        Parameters
        ----------
        img : ndarray, pyvips.Image
            Image to warp. Should be a scaled version of the same one used for registration

        to_slide_obj : Slide
            Slide to which the points will be warped. I.e. `xy`
            will be warped from this Slide to their position in
            the unwarped slide associated with `to_slide_obj`.

        dst_slide_level: int, tuple, optional
            Pyramid level of the slide/image that `img` will be warped on to

        non_rigid : bool, optional
            Whether or not to conduct non-rigid warping. If False,
            then only a rigid transformation will be applied.

        """

        if np.issubdtype(type(dst_slide_level), np.integer):
            to_slide_src_shape_rc = to_slide_obj.slide_dimensions_wh[dst_slide_level][::-1]
            aligned_slide_shape = self.val_obj.get_aligned_slide_shape(dst_slide_level)
        else:

            to_slide_src_shape_rc = np.array(dst_slide_level)

            dst_scale_rc = (to_slide_src_shape_rc/np.array(to_slide_obj.processed_img_shape_rc))
            aligned_slide_shape = np.round(dst_scale_rc*np.array(to_slide_obj.reg_img_shape_rc)).astype(int)

        if non_rigid:
            from_bk_dxdy = self.bk_dxdy
            to_fwd_dxdy = to_slide_obj.fwd_dxdy

        else:
            from_bk_dxdy = None
            to_fwd_dxdy = None

        warped_img = \
            warp_tools.warp_img_from_to(img,
                                        from_M=self.M,
                                        from_transformation_src_shape_rc=self.processed_img_shape_rc,
                                        from_transformation_dst_shape_rc=self.reg_img_shape_rc,
                                        from_dst_shape_rc=aligned_slide_shape,
                                        from_bk_dxdy=from_bk_dxdy,
                                        to_M=to_slide_obj.M,
                                        to_transformation_src_shape_rc=to_slide_obj.processed_img_shape_rc,
                                        to_transformation_dst_shape_rc=to_slide_obj.reg_img_shape_rc,
                                        to_src_shape_rc=to_slide_src_shape_rc,
                                        to_fwd_dxdy=to_fwd_dxdy,
                                        bg_color=bg_color,
                                        interp_method=interp_method
                                        )

        return warped_img

    @valtils.deprecated_args(crop_to_overlap="crop")
    def warp_slide(self, level, non_rigid=True, crop=True,
                   src_f=None, interp_method="bicubic", reader=None):
        """Warp a slide using registration parameters

        Parameters
        ----------
        level : int
            Pyramid level to be warped

        non_rigid : bool, optional
            Whether or not to conduct non-rigid warping. If False,
            then only a rigid transformation will be applied. Default is True

        crop: bool, str
            How to crop the registered images. If `True`, then the same crop used
            when initializing the `Valis` object will be used. If `False`, the
            image will not be cropped. If "overlap", the warped slide will be
            cropped to include only areas where all images overlapped.
            "reference" crops to the area that overlaps with the reference image,
            defined by `reference_img_f` when initialzing the `Valis object`.

        src_f : str, optional
           Path of slide to be warped. If None (the default), Slide.src_f
           will be used. Otherwise, the file to which `src_f` points to should
           be an alternative copy of the slide, such as one that has undergone
           processing (e.g. stain segmentation), has a mask applied, etc...

        interp_method : str
            Interpolation method used when warping slide. Default is "bicubic"

        """
        if src_f is None:
            src_f = self.src_f

        if non_rigid:
            bk_dxdy = self.bk_dxdy
        else:
            bk_dxdy = None

        if level != 0:
            if not np.issubdtype(type(level), np.integer):
                msg = "Need slide level to be an integer indicating pyramid level"
                valtils.print_warning(msg)
            aligned_slide_shape = self.val_obj.get_aligned_slide_shape(level)
        else:
            aligned_slide_shape = self.aligned_slide_shape_rc

        if isinstance(crop, bool) or isinstance(crop, str):
            crop_method = self.get_crop_method(crop)
            if crop_method is not False:
                if crop_method == CROP_REF:
                    ref_slide = self.val_obj.get_ref_slide()
                    scaled_aligned_shape_rc = ref_slide.slide_dimensions_wh[level][::-1]

                elif crop_method == CROP_OVERLAP:
                    scaled_aligned_shape_rc = aligned_slide_shape

                slide_bbox_xywh, _ = self.get_crop_xywh(crop=crop_method,
                                                        out_shape_rc=scaled_aligned_shape_rc)

                if crop_method == CROP_REF:
                    assert np.all(slide_bbox_xywh[2:] == scaled_aligned_shape_rc[::-1])
                    if src_f == self.src_f and self == ref_slide:
                        # Shouldn't need to warp, but do checks just in case
                        no_rigid = True
                        no_non_rigid = True
                        if self.M is not None:
                            sxy = (scaled_aligned_shape_rc/self.processed_img_shape_rc)[::-1]
                            scaled_txy = sxy*self.M[:2, 2]
                            no_transforms = all(self.M[:2, :2].reshape(-1) == [1, 0, 0, 1])
                            crop_to_origin = np.all(np.abs(slide_bbox_xywh[0:2] + scaled_txy) < 1)
                            no_rigid = no_transforms and crop_to_origin

                        if self.bk_dxdy is not None:
                            no_non_rigid = self.bk_dxdy.min() == 0 and self.bk_dxdy.max() == 0

                        if no_rigid and no_non_rigid:
                            # Don't need to warp, so return original reference image
                            ref_img, res = self.reader.slide2vips(level=level)
                            return ref_img
                        # else:
                        #     print("unexpectedly have to warp reference image. This may be due to an error")
            else:
                slide_bbox_xywh = None

        elif isinstance(crop[0], (int, float)) and len(crop) == 4:
            slide_bbox_xywh = crop
        else:
            slide_bbox_xywh = None

        if src_f == self.src_f:
            bg_color = self.bg_color
        else:
            bg_color = None

        if reader is None:
            reader = self.reader

        warped_slide = slide_tools.warp_slide(src_f, M=self.M,
                                              transformation_src_shape_rc=self.processed_img_shape_rc,
                                              transformation_dst_shape_rc=self.reg_img_shape_rc,
                                              aligned_slide_shape_rc=aligned_slide_shape,
                                              dxdy=bk_dxdy, level=level, series=self.series,
                                              interp_method=interp_method,
                                              bbox_xywh=slide_bbox_xywh,
                                              bg_color=bg_color,
                                              reader=reader)
        return warped_slide

    def warp_and_save_slide(self, dst_f, level=0, non_rigid=True,
                            crop=True, src_f=None,
                            channel_names=None,
                            colormap=slide_io.CMAP_AUTO,
                            interp_method="bicubic",
                            tile_wh=None, compression="lzw",
                            Q=100,
                            pyramid=True,
                            reader=None):

        """Warp and save a slide

        Slides will be saved in the ome.tiff format.

        Parameters
        ----------
        dst_f : str
            Path to were the warped slide will be saved.

        level : int
            Pyramid level to be warped

        non_rigid : bool, optional
            Whether or not to conduct non-rigid warping. If False,
            then only a rigid transformation will be applied. Default is True

        crop: bool, str
            How to crop the registered images. If `True`, then the same crop used
            when initializing the `Valis` object will be used. If `False`, the
            image will not be cropped. If "overlap", the warped slide will be
            cropped to include only areas where all images overlapped.
            "reference" crops to the area that overlaps with the reference image,
            defined by `reference_img_f` when initializing the `Valis object`.

        channel_names : list, optional
            List of channel names. If None, then Slide.reader
            will attempt to find the channel names associated with `src_f`.

        colormap : dict, optional
            Dictionary of channel colors, where the key is the channel name, and the value the color as rgb255.
            If None (default), the channel colors from `current_ome_xml_str` will be used, if available.
            If None, and there are no channel colors in the `current_ome_xml_str`, then no colors will be added

        src_f : str, optional
            Path of slide to be warped. If None (the default), Slide.src_f
            will be used. Otherwise, the file to which `src_f` points to should
            be an alternative copy of the slide, such as one that has undergone
            processing (e.g. stain segmentation), has a mask applied, etc...

        interp_method : str
            Interpolation method used when warping slide. Default is "bicubic"

        tile_wh : int, optional
            Tile width and height used to save image

        compression : str
            Compression method used to save ome.tiff . Default is lzw, but can also
            be jpeg or jp2k. See pyips for more details.

        Q : int
            Q factor for lossy compression

        pyramid : bool
            Whether or not to save an image pyramid.
        """

        if src_f is None:
            src_f = self.src_f

        if reader is None:
            if src_f != self.src_f:
                slide_reader_cls = slide_io.get_slide_reader(src_f)
                reader = slide_reader_cls(src_f)
            else:
                reader = self.reader

        warped_slide = self.warp_slide(level=level, non_rigid=non_rigid,
                                       crop=crop,
                                       interp_method=interp_method,
                                       src_f=src_f,
                                       reader=reader)

        # Get ome-xml #
        ome_xml_obj = slide_io.update_xml_for_new_img(img=warped_slide,
                                                      reader=reader,
                                                      level=level,
                                                      channel_names=channel_names,
                                                      colormap=colormap)

        ome_xml = ome_xml_obj.to_xml()

        out_shape_wh = warp_tools.get_shape(warped_slide)[0:2][::-1]
        tile_wh = slide_io.get_tile_wh(reader=reader,
                                       level=level,
                                       out_shape_wh=out_shape_wh)

        slide_io.save_ome_tiff(warped_slide, dst_f=dst_f, ome_xml=ome_xml,
                               tile_wh=tile_wh, compression=compression,
                               Q=Q, pyramid=pyramid)


    def warp_xy(self, xy, M=None, slide_level=0, pt_level=0,
                non_rigid=True, crop=True):
        """Warp points using registration parameters

        Warps `xy` to their location in the registered slide/image

        Parameters
        ----------
        xy : ndarray
            (N, 2) array of points to be warped. Must be x,y coordinates

        slide_level: int, tuple, optional
            Pyramid level of the slide. Used to scale transformation matrices.
            Can also be the shape of the warped image (row, col) into which
            the points should be warped. Default is 0.

        pt_level: int, tuple, optional
            Pyramid level from which the points origingated. For example, if
            `xy` are from the centroids of cell segmentation performed on the
            full resolution image, this should be 0. Alternatively, the value can
            be a tuple of the image's shape (row, col) from which the points came.
            For example, if `xy` are  bounding box coordinates from an analysis on
            a lower resolution image, then pt_level is that lower resolution
            image's shape (row, col). Default is 0.

        non_rigid : bool, optional
            Whether or not to conduct non-rigid warping. If False,
            then only a rigid transformation will be applied. Default is True.

        crop: bool, str
            Apply crop to warped points by shifting points to the mask's origin.
            Note that this can result in negative coordinates, but might be useful
            if wanting to draw the coordinates on the registered slide, such as
            annotation coordinates.

            If `True`, then the same crop used
            when initializing the `Valis` object will be used. If `False`, the
            image will not be cropped. If "overlap", the warped slide will be
            cropped to include only areas where all images overlapped.
            "reference" crops to the area that overlaps with the reference image,
            defined by `reference_img_f` when initialzing the `Valis object`.

        """
        if M is None:
            M = self.M

        if np.issubdtype(type(pt_level), np.integer):
            pt_dim_rc = self.slide_dimensions_wh[pt_level][::-1]
        else:
            pt_dim_rc = np.array(pt_level)

        if np.issubdtype(type(slide_level), np.integer):
            if slide_level != 0:
                if np.issubdtype(type(slide_level), np.integer):
                    aligned_slide_shape = self.val_obj.get_aligned_slide_shape(slide_level)
                else:
                    aligned_slide_shape = np.array(slide_level)
            else:
                aligned_slide_shape = self.aligned_slide_shape_rc
        else:
            aligned_slide_shape = np.array(slide_level)

        if non_rigid:
            fwd_dxdy = self.fwd_dxdy
        else:
            fwd_dxdy = None

        warped_xy = warp_tools.warp_xy(xy, M=M,
                                       transformation_src_shape_rc=self.processed_img_shape_rc,
                                       transformation_dst_shape_rc=self.reg_img_shape_rc,
                                       src_shape_rc=pt_dim_rc,
                                       dst_shape_rc=aligned_slide_shape,
                                       fwd_dxdy=fwd_dxdy)

        crop_method = self.get_crop_method(crop)
        if crop_method is not False:
            if crop_method == CROP_REF:
                ref_slide = self.val_obj.get_ref_slide()
                if isinstance(slide_level, int):
                    scaled_aligned_shape_rc = ref_slide.slide_dimensions_wh[slide_level][::-1]
                else:
                    if len(slide_level) == 2:
                        scaled_aligned_shape_rc = slide_level
            elif crop_method == CROP_OVERLAP:
                scaled_aligned_shape_rc = aligned_slide_shape

            crop_bbox_xywh, _ = self.get_crop_xywh(crop_method, scaled_aligned_shape_rc)
            warped_xy -= crop_bbox_xywh[0:2]

        return warped_xy

    def warp_xy_from_to(self, xy, to_slide_obj, src_slide_level=0, src_pt_level=0,
                        dst_slide_level=0, non_rigid=True):

        """Warp points from this slide to another unwarped slide

        Takes a set of points found in this unwarped slide, and warps them to
        their position in the unwarped "to" slide.

        Parameters
        ----------
        xy : ndarray
            (N, 2) array of points to be warped. Must be x,y coordinates

        to_slide_obj : Slide
            Slide to which the points will be warped. I.e. `xy`
            will be warped from this Slide to their position in
            the unwarped slide associated with `to_slide_obj`.

        src_pt_level: int, tuple, optional
            Pyramid level of the slide/image in which `xy` originated.
            For example, if `xy` are from the centroids of cell segmentation
            performed on the unwarped full resolution image, this should be 0.
            Alternatively, the value can be a tuple of the image's shape (row, col)
            from which the points came. For example, if `xy` are  bounding
            box coordinates from an analysis on a lower resolution image,
            then pt_level is that lower resolution image's shape (row, col).

        dst_slide_level: int, tuple, optional
            Pyramid level of the slide/image in to `xy` will be warped.
            Similar to `src_pt_level`, if `dst_slide_level` is an int then
            the points will be warped to that pyramid level. If `dst_slide_level`
            is the "to" image's shape (row, col), then the points will be warped
            to their location in an image with that same shape.

        non_rigid : bool, optional
            Whether or not to conduct non-rigid warping. If False,
            then only a rigid transformation will be applied.

        """

        if np.issubdtype(type(src_pt_level), np.integer):
            src_pt_dim_rc = self.slide_dimensions_wh[src_pt_level][::-1]
        else:
            src_pt_dim_rc = np.array(src_pt_level)

        if np.issubdtype(type(dst_slide_level), np.integer):
            to_slide_src_shape_rc = to_slide_obj.slide_dimensions_wh[dst_slide_level][::-1]
        else:
            to_slide_src_shape_rc = np.array(dst_slide_level)

        if src_slide_level != 0:
            if np.issubdtype(type(src_slide_level), np.integer):
                aligned_slide_shape = self.val_obj.get_aligned_slide_shape(src_slide_level)
            else:
                aligned_slide_shape = np.array(src_slide_level)
        else:
            aligned_slide_shape = self.aligned_slide_shape_rc

        if non_rigid:
            src_fwd_dxdy = self.fwd_dxdy
            dst_bk_dxdy = to_slide_obj.bk_dxdy

        else:
            src_fwd_dxdy = None
            dst_bk_dxdy = None

        xy_in_unwarped_to_img = \
            warp_tools.warp_xy_from_to(xy=xy,
                                       from_M=self.M,
                                       from_transformation_dst_shape_rc=self.reg_img_shape_rc,
                                       from_transformation_src_shape_rc=self.processed_img_shape_rc,
                                       from_dst_shape_rc=aligned_slide_shape,
                                       from_src_shape_rc=src_pt_dim_rc,
                                       from_fwd_dxdy=src_fwd_dxdy,
                                       to_M=to_slide_obj.M,
                                       to_transformation_src_shape_rc=to_slide_obj.processed_img_shape_rc,
                                       to_transformation_dst_shape_rc=to_slide_obj.reg_img_shape_rc,
                                       to_src_shape_rc=to_slide_src_shape_rc,
                                       to_dst_shape_rc=aligned_slide_shape,
                                       to_bk_dxdy=dst_bk_dxdy
                                       )

        return xy_in_unwarped_to_img

    def warp_geojson(self, geojson_f, M=None, slide_level=0, pt_level=0,
                non_rigid=True, crop=True):
        """Warp geometry using registration parameters

        Warps geometries to their location in the registered slide/image

        Parameters
        ----------
        geojson_f : str
            Path to geojson file containing the annotation geometries. Assumes
            coordinates are in pixels.

        slide_level: int, tuple, optional
            Pyramid level of the slide. Used to scale transformation matrices.
            Can also be the shape of the warped image (row, col) into which
            the points should be warped. Default is 0.

        pt_level: int, tuple, optional
            Pyramid level from which the points origingated. For example, if
            `xy` are from the centroids of cell segmentation performed on the
            full resolution image, this should be 0. Alternatively, the value can
            be a tuple of the image's shape (row, col) from which the points came.
            For example, if `xy` are  bounding box coordinates from an analysis on
            a lower resolution image, then pt_level is that lower resolution
            image's shape (row, col). Default is 0.

        non_rigid : bool, optional
            Whether or not to conduct non-rigid warping. If False,
            then only a rigid transformation will be applied. Default is True.

        crop: bool, str
            Apply crop to warped points by shifting points to the mask's origin.
            Note that this can result in negative coordinates, but might be useful
            if wanting to draw the coordinates on the registered slide, such as
            annotation coordinates.

            If `True`, then the same crop used
            when initializing the `Valis` object will be used. If `False`, the
            image will not be cropped. If "overlap", the warped slide will be
            cropped to include only areas where all images overlapped.
            "reference" crops to the area that overlaps with the reference image,
            defined by `reference_img_f` when initialzing the `Valis object`.

        """
        if M is None:
            M = self.M

        if np.issubdtype(type(pt_level), np.integer):
            pt_dim_rc = self.slide_dimensions_wh[pt_level][::-1]
        else:
            pt_dim_rc = np.array(pt_level)

        if np.issubdtype(type(slide_level), np.integer):
            if slide_level != 0:
                if np.issubdtype(type(slide_level), np.integer):
                    aligned_slide_shape = self.val_obj.get_aligned_slide_shape(slide_level)
                else:
                    aligned_slide_shape = np.array(slide_level)
            else:
                aligned_slide_shape = self.aligned_slide_shape_rc
        else:
            aligned_slide_shape = np.array(slide_level)

        if non_rigid:
            fwd_dxdy = self.fwd_dxdy
        else:
            fwd_dxdy = None

        with open(geojson_f) as f:
            annotation_geojson = json.load(f)

        crop_method = self.get_crop_method(crop)
        if crop_method is not False:
            if crop_method == CROP_REF:
                ref_slide = self.val_obj.get_ref_slide()
                if isinstance(slide_level, int):
                    scaled_aligned_shape_rc = ref_slide.slide_dimensions_wh[slide_level][::-1]
                else:
                    if len(slide_level) == 2:
                        scaled_aligned_shape_rc = slide_level
            elif crop_method == CROP_OVERLAP:
                scaled_aligned_shape_rc = aligned_slide_shape

            crop_bbox_xywh, _ = self.get_crop_xywh(crop_method, scaled_aligned_shape_rc)
            shift_xy = crop_bbox_xywh[0:2]
        else:
            shift_xy = None

        warped_features = [None]*len(annotation_geojson["features"])
        for i, ft in tqdm.tqdm(enumerate(annotation_geojson["features"]), desc=WARP_ANNO_MSG, unit="annotation"):
            geom = shapely.geometry.shape(ft["geometry"])
            warped_geom = warp_tools.warp_shapely_geom(geom, M=M,
                                            transformation_src_shape_rc=self.processed_img_shape_rc,
                                            transformation_dst_shape_rc=self.reg_img_shape_rc,
                                            src_shape_rc=pt_dim_rc,
                                            dst_shape_rc=aligned_slide_shape,
                                            fwd_dxdy=fwd_dxdy,
                                            shift_xy=shift_xy)
            warped_ft = deepcopy(ft)
            warped_ft["geometry"] = shapely.geometry.mapping(warped_geom)
            warped_features[i] = warped_ft

        warped_geojson = {"type":annotation_geojson["type"], "features":warped_features}

        return warped_geojson

    def warp_geojson_from_to(self, geojson_f, to_slide_obj, src_slide_level=0, src_pt_level=0,
                            dst_slide_level=0, non_rigid=True):
        """Warp geoms in geojson file from annotation slide to another unwarped slide

        Takes a set of geometries found in this annotation slide, and warps them to
        their position in the unwarped "to" slide.

        Parameters
        ----------
        geojson_f : str
            Path to geojson file containing the annotation geometries. Assumes
            coordinates are in pixels.

        to_slide_obj : Slide
            Slide to which the points will be warped. I.e. `xy`
            will be warped from this Slide to their position in
            the unwarped slide associated with `to_slide_obj`.

        src_pt_level: int, tuple, optional
            Pyramid level of the slide/image in which `xy` originated.
            For example, if `xy` are from the centroids of cell segmentation
            performed on the unwarped full resolution image, this should be 0.
            Alternatively, the value can be a tuple of the image's shape (row, col)
            from which the points came. For example, if `xy` are  bounding
            box coordinates from an analysis on a lower resolution image,
            then pt_level is that lower resolution image's shape (row, col).

        dst_slide_level: int, tuple, optional
            Pyramid level of the slide/image in to `xy` will be warped.
            Similar to `src_pt_level`, if `dst_slide_level` is an int then
            the points will be warped to that pyramid level. If `dst_slide_level`
            is the "to" image's shape (row, col), then the points will be warped
            to their location in an image with that same shape.

        non_rigid : bool, optional
            Whether or not to conduct non-rigid warping. If False,
            then only a rigid transformation will be applied.

        Returns
        -------
        warped_geojson : dict
            Dictionry of warped geojson geometries

        """

        if np.issubdtype(type(src_pt_level), np.integer):
            src_pt_dim_rc = self.slide_dimensions_wh[src_pt_level][::-1]
        else:
            src_pt_dim_rc = np.array(src_pt_level)

        if np.issubdtype(type(dst_slide_level), np.integer):
            to_slide_src_shape_rc = to_slide_obj.slide_dimensions_wh[dst_slide_level][::-1]
        else:
            to_slide_src_shape_rc = np.array(dst_slide_level)

        if src_slide_level != 0:
            if np.issubdtype(type(src_slide_level), np.integer):
                aligned_slide_shape = self.val_obj.get_aligned_slide_shape(src_slide_level)
            else:
                aligned_slide_shape = np.array(src_slide_level)
        else:
            aligned_slide_shape = self.aligned_slide_shape_rc

        if non_rigid:
            src_fwd_dxdy = self.fwd_dxdy
            dst_bk_dxdy = to_slide_obj.bk_dxdy

        else:
            src_fwd_dxdy = None
            dst_bk_dxdy = None

        with open(geojson_f) as f:
            annotation_geojson = json.load(f)

        warped_features = [None]*len(annotation_geojson["features"])
        for i, ft in tqdm.tqdm(enumerate(annotation_geojson["features"]), desc=WARP_ANNO_MSG, unit="annotation"):
            geom = shapely.geometry.shape(ft["geometry"])
            warped_geom = warp_tools.warp_shapely_geom_from_to(geom=geom,
                                            from_M=self.M,
                                            from_transformation_dst_shape_rc=self.reg_img_shape_rc,
                                            from_transformation_src_shape_rc=self.processed_img_shape_rc,
                                            from_dst_shape_rc=aligned_slide_shape,
                                            from_src_shape_rc=src_pt_dim_rc,
                                            from_fwd_dxdy=src_fwd_dxdy,
                                            to_M=to_slide_obj.M,
                                            to_transformation_src_shape_rc=to_slide_obj.processed_img_shape_rc,
                                            to_transformation_dst_shape_rc=to_slide_obj.reg_img_shape_rc,
                                            to_src_shape_rc=to_slide_src_shape_rc,
                                            to_dst_shape_rc=aligned_slide_shape,
                                            to_bk_dxdy=dst_bk_dxdy
                                            )

            warped_ft = deepcopy(ft)
            warped_ft["geometry"] = shapely.geometry.mapping(warped_geom)
            warped_features[i] = warped_ft

        warped_geojson = {"type":annotation_geojson["type"], "features":warped_features}

        return warped_geojson


class Valis(object):
    """Reads, registers, and saves a series of slides/images

    Implements the registration pipeline described in
    "VALIS: Virtual Alignment of pathoLogy Image Series" by Gatenbee et al.
    This pipeline will read images and whole slide images (WSI) using pyvips,
    bioformats, or openslide, and so should work with a wide variety of formats.
    VALIS can perform both rigid and non-rigid registration. The registered slides
    can be saved as ome.tiff slides that can be used in downstream analyses. The
    ome.tiff format is opensource and widely supported, being readable in several
    different programming languages (Python, Java, Matlab, etc...) and software,
    such as QuPath or HALO.

    The pipeline is fully automated and goes as follows:

    1. Images/slides are converted to numpy arrays. As WSI are often
    too large to fit into memory, these images are usually lower resolution
    images from different pyramid levels.

    2. Images are processed to single channel images. They are then
    normalized to make them look as similar as possible.

    3. Image features are detected and then matched between all pairs of image.

    4. If the order of images is unknown, they will be optimally ordered
    based on their feature similarity

    5. Rigid registration is performed serially, with each image being
    rigidly aligned to the previous image in the stack.

    6. Non-rigid registration is then performed either by 1) aliging each image
    towards the center of the stack, composing the deformation fields
    along the way, or 2) using groupwise registration that non-rigidly aligns
    the images to a common frame of reference.

    7. Error is measured by calculating the distance between registered
    matched features.

    The transformations found by VALIS can then be used to warp the full
    resolution slides. It is also possible to merge non-RGB registered slides
    to create a highly multiplexed image. These aligned and/or merged slides
    can then be saved as ome.tiff images using pyvips.

    In addition to warping images and slides, VALIS can also warp point data,
    such as cell centoids or ROI coordinates.

    Attributes
    ----------
    name : str
        Descriptive name of registrar, such as the sample's name.

    src_dir: str
        Path to directory containing the slides that will be registered.

    dst_dir : str
        Path to where the results should be saved.

    original_img_list : list of ndarray
        List of images converted from the slides in `src_dir`

    name_dict : dictionary
        Key=full path to image, value = name used to look up `Slide` in `Valis.slide_dict`

    slide_dims_dict_wh :
        Dictionary of slide dimensions. Only needed if dimensions not
        available in the slide/image's metadata.

    resolution_xyu: tuple
        Physical size per pixel and the unit.

    image_type : str
        Type of image, i.e. "brightfield" or "fluorescence"

    series : int
        Slide series to that was read.

    size : int
        Number of images to align

    aligned_img_shape_rc : tuple of int
        Shape (row, col) of aligned images

    aligned_slide_shape_rc : tuple of int
        Shape (row, col) of the aligned slides

    slide_dict : dict of Slide
        Dictionary of Slide objects, each of which contains information
        about a slide, and methods to warp it.

    brightfield_procsseing_fxn_str: str
        Name of function used to process brightfield images.

    if_procsseing_fxn_str : str
        Name of function used to process fluorescence images.

    max_image_dim_px : int
        Maximum width or height of images that will be saved.
        This limit is mostly to keep memory in check.

    max_processed_image_dim_px : int
        Maximum width or height of processed images. An important
        parameter, as it determines the size of of the image in which
        features will be detected and displacement fields computed.

    reference_img_f : str
        Filename of image that will be treated as the center of the stack.
        If None, the index of the middle image will be the reference.

    reference_img_idx : int
        Index of slide that corresponds to `reference_img_f`, after
        the `img_obj_list` has been sorted during rigid registration.

    align_to_reference : bool
        Whether or not images should be aligne to a reference image
        specified by `reference_img_f`. Will be set to True if
        `reference_img_f` is provided.

    crop: str, optional
        How to crop the registered images.

    rigid_registrar : SerialRigidRegistrar
        SerialRigidRegistrar object that performs the rigid registration.

    rigid_reg_kwargs : dict
        Dictionary of keyward arguments passed to
        `serial_rigid.register_images`.

    feature_descriptor_str : str
        Name of feature descriptor.

    feature_detector_str : str
        Name of feature detector.

    transform_str : str
        Name of rigid transform

    similarity_metric : str
        Name of similarity metric used to order slides.

    match_filter_method : str
        Name of method used to filter out poor feature matches.

    non_rigid_registrar : SerialNonRigidRegistrar
        SerialNonRigidRegistrar object that performs serial
        non-rigid registration.

    non_rigid_reg_kwargs : dict
        Dictionary of keyward arguments passed to
        `serial_non_rigid.register_images`.

    non_rigid_registrar_cls : NonRigidRegistrar
        Uninstantiated NonRigidRegistrar class that will be used
        by `non_rigid_registrar` to calculate the deformation fields
        between images.

    non_rigid_reg_class_str : str
        Name of the of class `non_rigid_registrar_cls` belongs to.

    thumbnail_size : int
        Maximum width or height of thumbnails that show results

    original_overlap_img : ndarray
        Image showing how original images overlap before registration.
        Created by merging coloring the inverted greyscale copies of each
        image, and then merging those images.

    rigid_overlap_img : ndarray
        Image showing how images overlap after rigid registration.

    non_rigid_overlap_img : ndarray
        Image showing how images overlap after rigid + non-rigid registration.

    has_rounds : bool
        Whether or not the contents of `src_dir` contain subdirectories that
        have single images spread across multiple files. An example would be
        .ndpis images.

    norm_method : str
        Name of method used to normalize the processed images

    target_processing_stats : ndarray
        Array of processed images' stats used to normalize all images

    summary_df : pd.Dataframe
        Pandas dataframe containing information about the results, such
        as the error, shape of aligned slides, time to completion, etc...

    start_time : float
        The time at which registation was initiated.

    end_rigid_time : float
        The time at which rigid registation was completed.

    end_non_rigid_time : float
        The time at which non-rigid registation was completed.

    qt_emitter : PySide2.QtCore.Signal
        Used to emit signals that update the GUI's progress bars

    _non_rigid_bbox : list
        Bounding box of area in which non-rigid registration was conducted

    _full_displacement_shape_rc : tuple
        Shape of full displacement field. Would be larger than `_non_rigid_bbox`
        if non-rigid registration only performed in a masked region

    _dup_names_dict : dictionary
        Dictionary describing which images would have been assigned duplicate
        names. Key= duplicated name, value=list of paths to images which
        would have been assigned the same name

    _empty_slides : dictionary
        Dictionary of `Slide` objects that have empty images. Ignored during
        registration but added back at the end


    Examples
    --------

    Basic example using default parameters

    >>> from valis import registration, data
    >>> slide_src_dir = data.dcis_src_dir
    >>> results_dst_dir = "./slide_registration_example"
    >>> registered_slide_dst_dir = "./slide_registration_example/registered_slides"

    Perform registration

    >>> rigid_registrar, non_rigid_registrar, error_df = registrar.register()

    View results in "./slide_registration_example".
    If they look good, warp and save the slides as ome.tiff

    >>> registrar.warp_and_save_slides(registered_slide_dst_dir)

    This example shows how to register CyCIF images and then merge
    to create a high dimensional ome.tiff slide

    >>> registrar = registration.Valis(slide_src_dir, results_dst_dir)
    >>> rigid_registrar, non_rigid_registrar, error_df = registrar.register()

    Create function to get marker names from each slides' filename

    >>> def cnames_from_filename(src_f):
    ...     f = valtils.get_name(src_f)
    ...     return ["DAPI"] + f.split(" ")[1:4]
    ...
    >>> channel_name_dict = {f:cnames_from_filename(f) for f in  registrar.original_img_list}
    >>> merged_img, channel_names, ome_xml = registrar.warp_and_merge_slides(merged_slide_dst_f, channel_name_dict=channel_name_dict)

    View ome.tiff, located at merged_slide_dst_f

    """
    @valtils.deprecated_args(max_non_rigid_registartion_dim_px="max_non_rigid_registration_dim_px", img_type="image_type")
    def __init__(self, src_paths, dst_dir, series=None, name=None, image_type=None,
                 feature_detector_cls=DEFAULT_FD,
                 transformer_cls=DEFAULT_TRANSFORM_CLASS,
                 affine_optimizer_cls=DEFAULT_AFFINE_OPTIMIZER_CLASS,
                 similarity_metric=DEFAULT_SIMILARITY_METRIC,
                 matcher=DEFAULT_MATCH_FILTER,
                 imgs_ordered=True, #Normalmente va a False, pero para inizializar mejor a True
                 non_rigid_registrar_cls=DEFAULT_NON_RIGID_CLASS,
                 non_rigid_reg_params=DEFAULT_NON_RIGID_KWARGS,
                 compose_non_rigid=False,
                 img_list=None,
                 reference_img_f=None,
                 align_to_reference=False,
                 do_rigid=True,
                 crop=None,
                 create_masks=True,
                 denoise_rigid=True,
                 check_for_reflections=False,
                 resolution_xyu=None,
                 slide_dims_dict_wh=None,
                 max_image_dim_px=DEFAULT_MAX_IMG_DIM,
                 max_processed_image_dim_px=DEFAULT_MAX_PROCESSED_IMG_SIZE,
                 max_non_rigid_registration_dim_px=DEFAULT_MAX_PROCESSED_IMG_SIZE,
                 thumbnail_size=DEFAULT_THUMBNAIL_SIZE,
                 norm_method=DEFAULT_NORM_METHOD,
                 micro_rigid_registrar_cls=None,
                 micro_rigid_registrar_params={},
                 qt_emitter=None,
                 img_etiquetes_he=None):

        """
        src_dir: str
            Path to directory containing the slides that will be registered.

        dst_dir : str
            Path to where the results should be saved.

        name : str, optional
            Descriptive name of registrar, such as the sample's name

        series : int, optional
            Slide series to that was read. If None, series will be set to 0.

        image_type : str, optional
            The type of image, either "brightfield", "fluorescence",
            or "multi". If None, VALIS will guess `image_type`
            of each image, based on the number of channels and datatype.
            Will assume that RGB = "brightfield",
            otherwise `image_type` will be set to "fluorescence".

        feature_detector_cls : FeatureDD, optional
            Uninstantiated FeatureDD object that detects and computes
            image features. Default is VggFD. The
            available feature_detectors are found in the `feature_detectors`
            module. If a desired feature detector is not available,
            one can be created by subclassing `feature_detectors.FeatureDD`.

        transformer_cls : scikit-image Transform class, optional
            Uninstantiated scikit-image transformer used to find
            transformation matrix that will warp each image to the target
            image. Default is SimilarityTransform

        affine_optimizer_cls : AffineOptimzer class, optional
            Uninstantiated AffineOptimzer that will minimize a
            cost function to find the optimal affine transformations.
            If a desired affine optimization is not available,
            one can be created by subclassing `affine_optimizer.AffineOptimizer`.

        similarity_metric : str, optional
            Metric used to calculate similarity between images, which is in
            turn used to build the distance matrix used to sort the images.
            Can be "n_matches", or a string to used as
            distance in spatial.distance.cdist. "n_matches"
            is the number of matching features between image pairs.

        match_filter_method: str, optional
            "GMS" will use filter_matches_gms() to remove poor matches.
            This uses the Grid-based Motion Statistics (GMS) or RANSAC.

        imgs_ordered : bool, optional
            Boolean defining whether or not the order of images in img_dir
            are already in the correct order. If True, then each filename should
            begin with the number that indicates its position in the z-stack. If
            False, then the images will be sorted by ordering a feature distance
            matix. Default is False.

        reference_img_f : str, optional
            Filename of image that will be treated as the center of the stack.
            If None, the index of the middle image will be the reference.

        align_to_reference : bool, optional
            If `False`, images will be non-rigidly aligned serially towards the
            reference image. If `True`, images will be non-rigidly aligned
            directly to the reference image. If `reference_img_f` is None,
            then the reference image will be the one in the middle of the stack.

        non_rigid_registrar_cls : NonRigidRegistrar, optional
            Uninstantiated NonRigidRegistrar class that will be used to
            calculate the deformation fields between images. See
            the `non_rigid_registrars` module for a desciption of available
            methods. If a desired non-rigid registration method is not available,
            one can be implemented by subclassing.NonRigidRegistrar.
            If None, then only rigid registration will be performed

        non_rigid_reg_params: dictionary, optional
            Dictionary containing key, value pairs to be used to initialize
            `non_rigid_registrar_cls`.
            In the case where simple ITK is used by the, params should be
            a SimpleITK.ParameterMap. Note that numeric values nedd to be
            converted to strings. See the NonRigidRegistrar classes in
            `non_rigid_registrars` for the available non-rigid registration
            methods and arguments.

        compose_non_rigid : bool, optional
            Whether or not to compose non-rigid transformations. If `True`,
            then an image is non-rigidly warped before aligning to the
            adjacent non-rigidly aligned image. This allows the transformations
            to accumulate, which may bring distant features together but could
            also  result  in un-wanted deformations, particularly around the edges.
            If `False`, the image not warped before being aaligned to the adjacent
            non-rigidly aligned image. This can reduce unwanted deformations, but
            may not bring distant features together.

        img_list : list, dictionary, optional
            List of images to be registered. However, it can also be a dictionary,
            in which case the key: value pairs are full_path_to_image: name_of_image,
            where name_of_image is the key that can be used to access the image from
            Valis.slide_dict.

        do_rigid: bool, dictionary, optional
            Whether or not to perform rigid registration. If `False`, rigid
            registration will be skipped.

            If `do_rigid` is a dictionary, it should contain inverse transformation
            matrices to rigidly align images to the specificed by `reference_img_f`.
            M will be estimated for images that are not in the dictionary.
            Each key is the filename of the image associated with the transformation matrix,
            and value is a dictionary containing the following values:
                `M` : (required) a 3x3 inverse transformation matrix as a numpy array.
                      Found by determining how to align fixed to moving.
                      If `M` was found by determining how to align moving to fixed,
                      then `M` will need to be inverted first.
                `transformation_src_shape_rc` : (optional) shape (row, col) of image used to find the rigid transformation.
                      If not provided, then it is assumed to be the shape of the level 0 slide
                `transformation_dst_shape_rc` : (optional) shape of registered image.
                      If not provided, this is assumed to be the shape of the level 0 reference slide.

        crop: str, optional
            How to crop the registered images. "overlap" will crop to include
            only areas where all images overlapped. "reference" crops to the
            area that overlaps with a reference image, defined by
            `reference_img_f`. This option can be used even if `reference_img_f`
            is `None` because the reference image will be set as the one at the center
            of the stack.

            If both `crop` and `reference_img_f` are `None`, `crop`
            will be set to "overlap". If `crop` is None, but `reference_img_f`
            is defined, then `crop` will be set to "reference".

        create_masks : bool, optional
            Whether or not to create and apply masks for registration.
            Can help focus alignment on the tissue, but can sometimes
            mask too much if there is a lot of variation in the image.

        denoise_rigid : bool, optional
            Whether or not to denoise processed images before rigid registion.
            Note that un-denoised images are used in the non-rigid registration

        check_for_reflections : bool, optional
            Determine if alignments are improved by relfecting/mirroring/flipping
            images. Optional because it requires re-detecting features in each version
            of the images and then re-matching features, and so can be time consuming and
            not always necessary.

        resolution_xyu: tuple, optional
            Physical size per pixel and the unit. If None (the default), these
            values will be determined for each slide using the slides' metadata.
            If provided, this physical pixel sizes will be used for all of the slides.
            This option is available in case one cannot easily access to the original
            slides, but does have the information on pixel's physical units.

        slide_dims_dict_wh : dict, optional
            Key= slide/image file name,
            value= dimensions = [(width, height), (width, height), ...] for each level.
            If None (the default), the slide dimensions will be pulled from the
            slides' metadata. If provided, those values will be overwritten. This
            option is available in case one cannot easily access to the original
            slides, but does have the information on the slide dimensions.

        max_image_dim_px : int, optional
            Maximum width or height of images that will be saved.
            This limit is mostly to keep memory in check.

        max_processed_image_dim_px : int, optional
            Maximum width or height of processed images. An important
            parameter, as it determines the size of of the image in which
            features will be detected and displacement fields computed.

        max_non_rigid_registration_dim_px : int, optional
             Maximum width or height of images used for non-rigid registration.
             Larger values may yeild more accurate results, at the expense of
             speed and memory. There is also a practical limit, as the specified
             size may be too large to fit in memory.

        mask_dict : dictionary
            Dictionary where key = overlap type (all, overlap, or reference), and
            value = (mask, mask_bbox_xywh)

        thumbnail_size : int, optional
            Maximum width or height of thumbnails that show results

        norm_method : str
            Name of method used to normalize the processed images. Options
            are None when normalization is not desired, "histo_match" for
            histogram matching and "img_stats" for normalizing by image statistics.
            See preprocessing.match_histograms and preprocessing.norm_khan
            for details.

        iter_order : list of tuples
            Each element of `iter_order` contains a tuple of stack
            indices. The first value is the index of the moving/current/from
            image, while the second value is the index of the moving/next/to
            image.

        micro_rigid_registrar_cls : MicroRigidRegistrar, optional
            Class used to perform higher resolution rigid registration. If `None`,
            this step is skipped.

        micro_rigid_registrar_params : dictionary
            Dictionary of keyword arguments used intialize the `MicroRigidRegistrar`

        qt_emitter : PySide2.QtCore.Signal, optional
            Used to emit signals that update the GUI's progress bars

        """
        '''
        # Get name, based on src directory
        if name is None:
            if src_dir.endswith(os.path.sep):
                name = os.path.split(src_dir[:-1])[1]
            else:
                name = os.path.split(src_dir)[1]
        self.name = name.replace(" ", "_")
        '''
        
        self.src_paths = src_paths
        if name is None:
            self.name = self._get_common_name_from_paths(src_paths)
        else:
            self.name = name.replace(" ", "_")

        
        # Set paths #
        #self.src_dir = src_dir
        self.dst_dir = os.path.join(dst_dir, self.name)
        self.name_dict = None

        if img_list is not None:
            if isinstance(img_list, dict):
                # Key=original file name, value=name
                self.original_img_list = list(img_list.keys())
                self.name_dict = img_list
            elif hasattr(img_list, "__iter__"):
                self.original_img_list = list(img_list)
            else:
                msg = (f"Cannot upack `img_list`, which is type {type(img_list).__name__}. "
                       "Please provide an iterable object (list, tuple, array, etc...) that has the location of the images")
                valtils.print_warning(msg, rgb=Fore.RED)
        else:
            self.get_imgs_in_dir()

        if self.name_dict is None:
            self.name_dict = self.get_img_names(self.original_img_list)

        self.check_for_duplicated_names(self.original_img_list)

        self.set_dst_paths()

        # Some information may already be provided #
        self.slide_dims_dict_wh = slide_dims_dict_wh
        self.resolution_xyu = resolution_xyu
        self.image_type = image_type

        # Results fields #
        self.series = series
        self.size = 0
        self.aligned_img_shape_rc = None
        self.aligned_slide_shape_rc = None
        self.slide_dict = {}

        # Fields related to image pre-processing #
        self.brightfield_procsseing_fxn_str = None
        self.if_procsseing_fxn_str = None

        if max_image_dim_px < max_processed_image_dim_px:
            msg = f"max_image_dim_px is {max_image_dim_px} but needs to be less or equal to {max_processed_image_dim_px}. Setting max_image_dim_px to {max_processed_image_dim_px}"
            valtils.print_warning(msg)
            max_image_dim_px = max_processed_image_dim_px

        self.max_image_dim_px = max_image_dim_px
        self.max_processed_image_dim_px = max_processed_image_dim_px
        self.max_non_rigid_registration_dim_px = max_non_rigid_registration_dim_px

        #Codi meu
        self.path_img_etiquetes_he = img_etiquetes_he
        self.img_etiquetes_he = None
        self.path_IHC_original = None

        # Setup rigid registration #
        self.reference_img_idx = None
        self.reference_img_f = reference_img_f
        self.align_to_reference = align_to_reference
        self.iter_order = None

        self.do_rigid = do_rigid
        self.rigid_registrar = None
        self.micro_rigid_registrar_cls = micro_rigid_registrar_cls
        self.micro_rigid_registrar_params = micro_rigid_registrar_params
        self.denoise_rigid = denoise_rigid

        self._set_rigid_reg_kwargs(name=name,
                                   feature_detector=feature_detector_cls,
                                   similarity_metric=similarity_metric,
                                   matcher=matcher,
                                   transformer=transformer_cls,
                                   affine_optimizer=affine_optimizer_cls,
                                   imgs_ordered=imgs_ordered,
                                   reference_img_f=reference_img_f,
                                   check_for_reflections=check_for_reflections,
                                   qt_emitter=qt_emitter)

        #Codi meu
        self.iou_rigid = None
        self.corr_rigid = None
        self.iou_non_rigid = None
        self.corr_non_rigid = None

        # Setup non-rigid registration #
        self.non_rigid_registrar = None
        self.non_rigid_registrar_cls = non_rigid_registrar_cls

        if crop is None:
            if reference_img_f is None:
                self.crop = CROP_OVERLAP
            else:
                self.crop = CROP_REF
        else:
            self.crop = crop

        self.compose_non_rigid = compose_non_rigid
        if non_rigid_registrar_cls is not None:
            self._set_non_rigid_reg_kwargs(name=name,
                                           non_rigid_reg_class=non_rigid_registrar_cls,
                                           non_rigid_reg_params=non_rigid_reg_params,
                                           reference_img_f=reference_img_f,
                                           compose_non_rigid=compose_non_rigid,
                                           qt_emitter=qt_emitter)

        # Info realted to saving images to view results #
        self.mask_dict = None
        self.create_masks = create_masks

        self.thumbnail_size = thumbnail_size
        self.original_overlap_img = None
        self.rigid_overlap_img = None
        self.non_rigid_overlap_img = None
        self.micro_reg_overlap_img = None

        self.has_rounds = False
        self.norm_method = norm_method
        self.summary_df = None
        self.start_time = None
        self.end_rigid_time = None
        self.end_non_rigid_time = None

        self._empty_slides = {}

    def _get_common_name_from_paths(self, paths):
        """
        Deducir el nombre común basado en los nombres de archivo en `paths`.
        Si no hay prefijo común, concatenar los nombres de archivo.
        """
        # Extraer solo los nombres de archivo sin extensiones
        filenames = [os.path.splitext(os.path.basename(path))[0] for path in paths]
        
        # Encontrar el prefijo común más largo
        common_prefix = os.path.commonprefix(filenames)

        if not common_prefix:
            # No hay prefijo común, concatenar los nombres sin extensiones
            common_prefix = "_".join(filenames)
        
        # Reemplazar cualquier espacio en el prefijo común
        common_prefix = common_prefix.replace(" ", "_")
        
        return common_prefix
    
    def _set_rigid_reg_kwargs(self, name, feature_detector, similarity_metric,
                              matcher, transformer, affine_optimizer,
                              imgs_ordered, reference_img_f, check_for_reflections, qt_emitter):

        """Set rigid registration kwargs
        Keyword arguments will be passed to `serial_rigid.register_images`

        """

        if affine_optimizer is not None:
            afo = affine_optimizer(transform=transformer.__name__)
        else:
            afo = affine_optimizer

        self.rigid_reg_kwargs = {NAME_KEY: name,
                                 FD_KEY: feature_detector(),
                                 SIM_METRIC_KEY: similarity_metric,
                                 TRANSFORMER_KEY: transformer(),
                                 MATCHER_KEY: matcher,
                                 AFFINE_OPTIMIZER_KEY: afo,
                                 REF_IMG_KEY: reference_img_f,
                                 IMAGES_ORDERD_KEY: imgs_ordered,
                                 CHECK_REFLECT_KEY: check_for_reflections,
                                 QT_EMMITER_KEY: qt_emitter
                                 }

        # Save methods as strings since some objects cannot be pickled #
        self.feature_descriptor_str = self.rigid_reg_kwargs[FD_KEY].kp_descriptor_name
        self.feature_detector_str = self.rigid_reg_kwargs[FD_KEY].kp_detector_name
        self.transform_str = self.rigid_reg_kwargs[TRANSFORMER_KEY].__class__.__name__
        self.similarity_metric = self.rigid_reg_kwargs[SIM_METRIC_KEY]
        self.match_filter_method = matcher.__class__.__name__
        self.imgs_ordered = imgs_ordered

    def _set_non_rigid_reg_kwargs(self, name, non_rigid_reg_class, non_rigid_reg_params,
                                  reference_img_f, compose_non_rigid, qt_emitter):
        """Set non-rigid registration kwargs
        Keyword arguments will be passed to `serial_non_rigid.register_images`

        """

        self.non_rigid_reg_kwargs = {NAME_KEY: name,
                                     NON_RIGID_REG_CLASS_KEY: non_rigid_reg_class,
                                     NON_RIGID_REG_PARAMS_KEY: non_rigid_reg_params,
                                     REF_IMG_KEY: reference_img_f,
                                     QT_EMMITER_KEY: qt_emitter,
                                     NON_RIGID_COMPOSE_KEY: compose_non_rigid
                                     }

        self.non_rigid_reg_class_str = self.non_rigid_reg_kwargs[NON_RIGID_REG_CLASS_KEY].__name__

    def _add_empty_slides(self):

        # Fill in missing attributes
        for slide_name, slide_obj in self._empty_slides.items():

            slide_obj.processed_img_shape_rc = slide_obj.image.shape[0:2]
            slide_obj.aligned_slide_shape_rc = self.aligned_slide_shape_rc
            slide_obj.reg_img_shape_rc = self.aligned_img_shape_rc

            slide_obj.processed_img = np.zeros(slide_obj.processed_img_shape_rc)
            slide_obj.rigid_reg_mask = np.full(slide_obj.processed_img_shape_rc, 255)
            slide_obj.non_rigid_reg_mask = np.full(slide_obj.reg_img_shape_rc, 255)

            slide_obj.M = np.eye(3)

            slide_obj.stack_idx = self.size
            self.size += 1
            self.slide_dict[slide_name] = slide_obj

    def get_imgs_in_dir(self):
        """Get all images in Valis.src_dir

        """
        #full_path_list = [os.path.join(self.src_dir, f) for f in os.listdir(self.src_dir)]
        full_path_list = self.src_paths
        self.original_img_list = []
        img_names = []
        for f in full_path_list:
            if os.path.isfile(f):
                if slide_tools.get_img_type(f) is not None:
                    self.original_img_list.append(f)
                    img_names.append(valtils.get_name(f))

        for f in full_path_list:
            if os.path.isdir(f):
                dir_name = os.path.split(f)[1]
                is_round, master_slide = slide_tools.determine_if_staining_round(f)
                if is_round:
                    self.original_img_list.append(master_slide)

                else:
                    # Some formats, like .mrxs have the main file but
                    # data in a subdirectory with the same name
                    matching_f = [ff for ff in full_path_list if re.search(dir_name, ff) is not None and os.path.split(ff)[1] != dir_name]
                    if len(matching_f) == 1:
                        if not matching_f[0] in self.original_img_list:
                            # Make sure that file not already in list
                            self.original_img_list.extend(matching_f)
                            img_names.append(dir_name)

                    elif len(matching_f) > 1:
                        msg = f"found {len(matching_f)} matches for {dir_name}: {', '.join(matching_f)}"
                        valtils.print_warning(msg, rgb=Fore.RED)
                    elif len(matching_f) == 0:
                        msg = f"Can't find slide file associated with {dir_name}"
                        valtils.print_warning(msg, rgb=Fore.RED)

    def set_dst_paths(self):
        """Set paths to where the results will be saved.

        """

        self.img_dir = os.path.join(self.dst_dir, CONVERTED_IMG_DIR)
        self.processed_dir = os.path.join(self.dst_dir, PROCESSED_IMG_DIR)
        self.reg_dst_dir = os.path.join(self.dst_dir, RIGID_REG_IMG_DIR)
        self.non_rigid_dst_dir = os.path.join(self.dst_dir, NON_RIGID_REG_IMG_DIR)
        self.deformation_field_dir = os.path.join(self.dst_dir, DEFORMATION_FIELD_IMG_DIR)
        self.overlap_dir = os.path.join(self.dst_dir, OVERLAP_IMG_DIR)
        self.data_dir = os.path.join(self.dst_dir, REG_RESULTS_DATA_DIR)
        self.displacements_dir = os.path.join(self.dst_dir, DISPLACEMENT_DIRS)
        self.micro_reg_dir = os.path.join(self.dst_dir, MICRO_REG_DIR)
        self.mask_dir = os.path.join(self.dst_dir, MASK_DIR)

        #Codi meu per crear el path a masks/Rigid
        subfolder_rigid = "Rigid"
        os.makedirs(os.path.join(self.mask_dir, subfolder_rigid), exist_ok=True)
        self.mask_dir_rigid = os.path.join(self.mask_dir, subfolder_rigid)
        #

        #Codi meu per crear el path a masks/Rigid/Registration
        subfolder_rigid_registration = "Registration"
        os.makedirs(os.path.join(self.mask_dir_rigid, subfolder_rigid_registration), exist_ok=True)
        self.mask_dir_rigid_registration = os.path.join(self.mask_dir_rigid, subfolder_rigid_registration)
        #

        #Codi meu per crear el path a masks/Rigid/Serial_rigid
        subfolder_rigid_serial_rigid = "Serial_rigid"
        os.makedirs(os.path.join(self.mask_dir_rigid, subfolder_rigid_serial_rigid), exist_ok=True)
        self.mask_dir_rigid_serial_rigid = os.path.join(self.mask_dir_rigid, subfolder_rigid_serial_rigid)
        #

        #Codi meu per crear el path a masks/Non_rigid
        subfolder_non_rigid = "Non_rigid"
        os.makedirs(os.path.join(self.mask_dir, subfolder_non_rigid), exist_ok=True)
        self.mask_dir_non_rigid = os.path.join(self.mask_dir, subfolder_non_rigid)
        #

        #Codi meu per crear el path a masks/Non_rigid/Registration
        subfolder_non_rigid_registration = "Registration"
        os.makedirs(os.path.join(self.mask_dir_non_rigid, subfolder_non_rigid_registration), exist_ok=True)
        self.mask_dir_non_rigid_registration = os.path.join(self.mask_dir_non_rigid, subfolder_non_rigid_registration)
        #

        #Codi meu per crear el path a masks/Non_rigid/Serial_non_rigid
        subfolder_non_rigid_serial_non_rigid = "Serial_non_rigid"
        os.makedirs(os.path.join(self.mask_dir_non_rigid, subfolder_non_rigid_serial_non_rigid), exist_ok=True)
        self.mask_dir_non_rigid_serial_non_rigid = os.path.join(self.mask_dir_non_rigid, subfolder_non_rigid_serial_non_rigid)
        #

        #Codi meu per crear el path a masks/Non_rigid/Non_rigid_registrars
        subfolder_non_rigid_non_rigid_registrars = "Non_rigid_registrars"
        os.makedirs(os.path.join(self.mask_dir_non_rigid, subfolder_non_rigid_non_rigid_registrars), exist_ok=True)
        self.mask_dir_non_rigid_non_rigid_registrars = os.path.join(self.mask_dir_non_rigid, subfolder_non_rigid_non_rigid_registrars)
        #

        #Codi meu per crear el path a masks/Preprocessing
        subfolder_preprocessing = "Preprocessing"
        os.makedirs(os.path.join(self.mask_dir, subfolder_preprocessing), exist_ok=True)
        self.mask_dir_preprocessing = os.path.join(self.mask_dir, subfolder_preprocessing)
        #



    def get_slide(self, src_f):
        """Get Slide

        Get the Slide associated with `src_f`.
        Slide store registration parameters and other metadata about
        the slide associated with `src_f`. Slide can also:

        * Convert the slide to a numpy array (Slide.slide2image)
        * Convert the slide to a pyvips.Image (Slide.slide2vips)
        * Warp the slide (Slide.warp_slide)
        * Save the warped slide as an ome.tiff (Slide.warp_and_save_slide)
        * Warp an image of the slide (Slide.warp_img)
        * Warp points (Slide.warp_xy)
        * Warp points in one slide to their position in another unwarped slide (Slide.warp_xy_from_to)
        * Access slide ome-xml (Slide.original_xml)

        See Slide for more details.

        Parameters
        ----------
        src_f : str
            Path to the slide, or name assigned to slide (see Valis.name_dict)

        Returns
        -------
        slide_obj : Slide
            Slide associated with src_f

        """

        default_name = valtils.get_name(src_f)

        if src_f in self.name_dict.keys():
            # src_f is full path to image
            assigned_name = self.name_dict[src_f]
        elif src_f in self.name_dict.values():
            # src_f is name of image
            assigned_name = src_f
        else:
            # src_f isn't in name_dict
            assigned_name = None

        
        if default_name in self.slide_dict:
            # src_f is the image name or file name
            slide_obj = self.slide_dict[default_name]

        elif assigned_name in self.slide_dict:
            # src_f is full path and name was looked up
            slide_obj = self.slide_dict[assigned_name]

        elif src_f in self.slide_dict:
            # src_f is the name of the slide
            slide_obj = self.slide_dict[src_f]

        elif default_name in self._dup_names_dict:
            # default name has multiple matches
            n_matching = len(self._dup_names_dict[default_name])
            possible_names_dict = {f: self.name_dict[f] for f in self._dup_names_dict[default_name]}

            msg = (f"\n{src_f} matches {n_matching} images in this dataset:\n"
                   f"{pformat(self._dup_names_dict[default_name])}"
                   f"\n\nPlease see `Valis.name_dict` to find correct name in "
                   f"the dictionary. Either key (filenmae) or value (assigned name) will work:\n"
                   f"{pformat(possible_names_dict)}")

            valtils.print_warning(msg, rgb=Fore.RED)
            slide_obj = None

        return slide_obj

    def get_ref_slide(self):
        ref_slide = self.get_slide(self.reference_img_f)

        return ref_slide

    def get_img_names(self, img_list):
        """
        Check that each image will have a unique name, which is based on the file name.
        Images that would otherwise have the same name are assigned extra ids, starting at 0.
        For example, if there were three images named "HE.tiff", they would be
        named "HE_0", "HE_1", and "HE_2".

        Parameters
        ----------

        img_list : list
            List of image names

        Returns
        -------
        name_dict : dict
            Dictionary, where key= full path to image, value = image name used as
            key in Valis.slide_dict

        """

        img_df = pd.DataFrame({"img_f": img_list,
                               "name": [valtils.get_name(f) for f in img_list]})

        names_dict = {f: valtils.get_name(f) for f in img_list}
        count_df = img_df["name"].value_counts()
        dup_idx = np.where(count_df.values > 1)[0]
        if len(dup_idx) > 0:
            for i in dup_idx:
                dup_name = count_df.index[i]
                dup_paths = img_df["img_f"][img_df["name"] == dup_name]
                z = len(str(len(dup_paths)))

                msg = f"Detected {len(dup_paths)} images that would be named {dup_name}"
                valtils.print_warning(msg, rgb=Fore.RED)

                for j, p in enumerate(dup_paths):
                    new_name = f"{names_dict[p]}_{str(j).zfill(z)}"
                    msg = f"Renmaing {p} to {new_name} in Valis.slide_dict)"
                    valtils.print_warning(msg)
                    names_dict[p] = new_name

        return names_dict

    def check_for_duplicated_names(self, img_list):
        """
        Create dictionary that tracks which files
        might be assigned the same name, which
        can happen if the filenames (minus the rest of the path) are the same
        """
        default_names_dict = {}
        for f in img_list:
            default_name = valtils.get_name(f)
            if default_name not in default_names_dict:
                default_names_dict[default_name] = [f]
            else:
                default_names_dict[default_name].append(f)

        self._dup_names_dict = {k: v for k, v in default_names_dict.items() if len(v) > 1}


    #Codi meu
    def png_to_vips_scaled(self, image, scaling):
        png_scaled = resize(image, (int(image.shape[0]*scaling), int(image.shape[1]*scaling)), preserve_range=True, anti_aliasing=True)
        png_scaled = rescale_intensity(png_scaled, in_range='image', out_range=(0, 1))
        png_scaled = img_as_ubyte(png_scaled)

        return png_scaled
    #

    def create_img_reader_dict(self, reader_dict=None, default_reader=None, series=None):

        if reader_dict is None:
            named_reader_dict = {}
        else:
            named_reader_dict = {valtils.get_name(f): reader_dict[f] for f in reader_dict.keys()}

        for i, slide_f in enumerate(self.original_img_list):
            slide_name = valtils.get_name(slide_f)
            if slide_name not in named_reader_dict:
                print(f'1')
                if default_reader is None:
                    print(f'2')
                    try:
                        slide_reader_cls = slide_io.get_slide_reader(slide_f, series=series)
                    except Exception as e:
                        traceback_msg = traceback.format_exc()
                        msg = f"Attempting to get reader for {slide_f} created the following error:\n{e}"
                        valtils.print_warning(msg, rgb=Fore.RED, traceback_msg=traceback_msg)
                else:
                    slide_reader_cls = default_reader

                slide_reader_kwargs = {"series": series}
            else:
                slide_reader_info = named_reader_dict[slide_name]
                if isinstance(slide_reader_info, list) or isinstance(slide_reader_info, tuple):
                    print(f'3')
                    if len(slide_reader_info) == 2:
                        print(f'4')
                        slide_reader_cls, slide_reader_kwargs = slide_reader_info
                    elif len(slide_reader_info) == 1:
                        print(f'5')
                        # Provided processor, but no kwargs
                        slide_reader_cls = slide_reader_info[0]
                        slide_reader_kwargs = {}
                else:
                    # Provided processor, but no kwargs
                    slide_reader_kwargs = {}
            try:
                slide_reader = slide_reader_cls(src_f=slide_f, **slide_reader_kwargs)
            except Exception as e:
                traceback_msg = traceback.format_exc()
                msg = f"Attempting to read {slide_f} created the following error:\n{e}"
                valtils.print_warning(msg, rgb=Fore.RED, traceback_msg=traceback_msg)

            named_reader_dict[slide_name] = slide_reader

        return named_reader_dict

    def convert_imgs(self, series=None, reader_dict=None, reader_cls=None):
        """Convert slides to images and create dictionary of Slides.
    
        series : int, optional
            Slide series to be read. If None, the series with largest image will be read
    
        reader_cls : SlideReader, optional
            Uninstantiated SlideReader class that will convert
            the slide to an image, and also collect metadata.
    
        reader_dict: dict, optional
            Dictionary specifying which readers to use for individual images.
            The keys, value pairs are image filename and instantiated `slide_io.SlideReader`
            to use to read that file. Valis will try to find an appropritate reader
            for any omitted files, or will use `reader_cls` as the default.
    
        """
    
        # Inicia el temporitzador general
        total_start_time = pytime.time()
    
        # Crear el diccionari de lectors
        start_time = pytime.time()
        named_reader_dict = self.create_img_reader_dict(reader_dict=reader_dict,
                                                        default_reader=reader_cls,
                                                        series=series)
        #print(f"Temps per crear el diccionari de lectors: {pytime.time() - start_time:.2f} segons")
    
        img_types = []
        self.size = 0
        #Codi meu
        i = 0
        #
        for f in tqdm.tqdm(self.original_img_list, desc=CONVERT_MSG, unit="image"):
            # Inicia temporitzador per a cada imatge
            slide_start_time = pytime.time()
    
            # Obté el nom de la diapositiva
            start_time = pytime.time()
            slide_name = valtils.get_name(f)
            #print(f"Temps per obtenir el nom: {pytime.time() - start_time:.2f} segons")
    
            # Lector associat
            start_time = pytime.time()
            reader = named_reader_dict[slide_name]
            slide_dims = reader.metadata.slide_dimensions
            #print(f'Dimensions de la slide: {slide_dims}')
            #print(f"Temps per carregar metadades del lector: {pytime.time() - start_time:.2f} segons")
    
            # Determina el nivell de detall
            start_time = pytime.time()
            levels_in_range = np.where(slide_dims.max(axis=1) < self.max_image_dim_px)[0]
            if len(levels_in_range) > 0:
                level = levels_in_range[0]
            else:
                level = len(slide_dims) - 1
            #print(f'Level:{level}')
            #print(f"Temps per seleccionar el nivell: {pytime.time() - start_time:.2f} segons")
    
            # Converteix la diapositiva a VIPS
            start_time = pytime.time()
            vips_img, path_img_original= reader.slide2vips(level=level)
            if i==0:
                path = os.path.join(self.mask_dir_preprocessing, "Lectura_imatge.png")
                warp_tools.save_img(path, vips_img)
            #print(f"Temps per convertir a VIPS: {pytime.time() - start_time:.2f} segons")
    
            # Escala si és necessari
            start_time = pytime.time()
            scaling = np.min(self.max_image_dim_px / np.array([vips_img.width, vips_img.height]))
            print(f'Factor de escalat:{scaling}')
            if scaling < 1:
                vips_img = warp_tools.rescale_img(vips_img, scaling)
                print(f'Entra')
            #print(f"Temps per escalar la imatge: {pytime.time() - start_time:.2f} segons")
    
            # Converteix VIPS a NumPy
            start_time = pytime.time()
            img = warp_tools.vips2numpy(vips_img)
            
            #Codi meu
            if i == 0:
                #vips_img_etiquetes = self.png_to_vips_scaled(np.array(img_etiquetes_he), scaling)
                #path = os.path.join(self.mask_dir_preprocessing, "Img_HE_etqiuetes_delmada_unpadding_padding.png")
                #warp_tools.save_img(path, vips_img_etiquetes)
                #self.conversio_mida_HE_etiquetes(path_HE_original = path_HE_original, img_etiquetes_he = img_etiquetes_he, level=level)
                            
                #Carregar la imatge d'etiquetes d'HE i treure-li el padding
                imatge_HE_etiquetes_sense_padding = self.eliminar_padding_calculat(path_HE_original = path_img_original, level = level)
                #
                self.preparar_imatge_HE_etiquetes(path_HE_original = path_img_original, imatge_HE_etiquetes_sense_padding = imatge_HE_etiquetes_sense_padding, level=level, rigid_or_non_rigid = 0)
            #

            if i == 1:  self.path_IHC_original = path_img_original


            #print(f"Temps per convertir a NumPy: {pytime.time() - start_time:.2f} segons")
    
            # Crea l'objecte Slide
            start_time = pytime.time()
            slide_obj = Slide(f, img, self, reader, name=slide_name)
            if i == 0:
                path = os.path.join(self.mask_dir_preprocessing, "Imatge_slide.png")
                warp_tools.save_img(path, slide_obj.image)
            slide_obj.crop = self.crop
            #print(f"Temps per crear l'objecte Slide: {pytime.time() - start_time:.2f} segons")
    
            # Will overwrite data if provided. Can occur if reading images, not the actual slides #
            if self.slide_dims_dict_wh is not None:
                print(f'1')
                start_time = pytime.time()
                matching_slide = [k for k in self.slide_dims_dict_wh.keys()
                                  if valtils.get_name(k) == slide_obj.name][0]
    
                slide_dims = self.slide_dims_dict_wh[matching_slide]
                if slide_dims.ndim == 1:
                    slide_dims = np.array([[slide_dims]])
                slide_obj.slide_shape_rc = slide_dims[0][::-1]
                #print(f"Temps per actualitzar dimensions del Slide: {pytime.time() - start_time:.2f} segons")
    
            if self.resolution_xyu is not None:
                print(f'2')
                slide_obj.resolution = np.mean(self.resolution_xyu[0:2])
                slide_obj.units = self.resolution_xyu[2]
    
            if slide_obj.is_empty:
                print(f'3')
                msg = f"{slide_obj.name} appears to be empty and will be skipped during registration"
                valtils.print_warning(msg)
                self._empty_slides[slide_obj.name] = slide_obj
                #print(f"Temps per manejar una imatge buida: {pytime.time() - slide_start_time:.2f} segons")
                continue
    
            # Afegir l'objecte Slide al diccionari
            start_time = pytime.time()
            img_types.append(slide_obj.img_type)
            self.slide_dict[slide_obj.name] = slide_obj
            self.size += 1
            #print(f"Temps per afegir la imatge al diccionari: {pytime.time() - start_time:.2f} segons")
    
            # Temps total per aquest slide
            #print(f"Temps total per {slide_name}: {pytime.time() - slide_start_time:.2f} segons")

            #Codi meu
            i = i+1
            #
    
        # Temps total de la funció
        print(f"Temps total per processar totes les imatges: {pytime.time() - total_start_time:.2f} segons")
    
        if self.image_type is None:
            unique_img_types = list(set(img_types))
            if len(unique_img_types) > 1:
                self.image_type = slide_tools.MULTI_MODAL_NAME
            else:
                self.image_type = unique_img_types[0]
    
        self.check_img_max_dims()


    def check_img_max_dims(self):
        """Ensure that all images have similar sizes.

        `max_image_dim_px` will be set to the maximum dimension of the
        smallest image if that value is less than max_image_dim_px

        """

        og_img_sizes_wh = np.array([slide_obj.image.shape[0:2][::-1] for slide_obj in self.slide_dict.values()])
        img_max_dims = og_img_sizes_wh.max(axis=1)
        min_max_wh = img_max_dims.min()
        scaling_for_og_imgs = min_max_wh/img_max_dims

        if np.any(scaling_for_og_imgs < 1):
            msg = f"Smallest image is less than max_image_dim_px. parameter max_image_dim_px is being set to {min_max_wh}"
            
            #Codi meu
            print(f"Smallest image is less than max_image_dim_px. parameter max_image_dim_px is being set to {min_max_wh}")
            #
            valtils.print_warning(msg)
            self.max_image_dim_px = min_max_wh
            for slide_obj in self.slide_dict.values():
                # Rescale images
                scaling = self.max_image_dim_px/max(slide_obj.image.shape[0:2])
                assert scaling <= self.max_image_dim_px
                if scaling < 1:
                    slide_obj.image = warp_tools.rescale_img(slide_obj.image, scaling)

        if self.max_processed_image_dim_px > self.max_image_dim_px:
            msg = f"parameter max_processed_image_dim_px also being updated to {self.max_image_dim_px}"
            
            #Codi meu
            print(f"parameter max_processed_image_dim_px also being updated to {self.max_image_dim_px}")
            #
            valtils.print_warning(msg)
            self.max_processed_image_dim_px = self.max_image_dim_px

        #Codi meu
        #print(f"Hola {self.max_image_dim_px}")
        return self.max_image_dim_px
        #
    
    def create_original_composite_img(self, rigid_registrar):
        """Create imaage showing how images overlap before registration
        """

        min_r = np.inf
        max_r = 0
        min_c = np.inf
        max_c = 0
        composite_img_list = [None] * self.size
        #Codi meu
        #imatge_a_reg_path = os.path.join(self.mask_dir_rigid_registration, "Màscara_teixit_HE_original.png")
        #imatge_a_reg = cv2.imread(imatge_a_reg_path)
        #
        for i, img_obj in enumerate(rigid_registrar.img_obj_list):
            img = img_obj.image
            
            #Codi meu
            #if i==0:
                #imatge_a_reg_path_1 = os.path.join(self.mask_dir_rigid_registration, "Imatge_teixit_HE_original_composite_image_1.png")
            #elif i == 1:
                #imatge_a_reg_path_1 = os.path.join(self.mask_dir_rigid_registration, "Imatge_teixit_HE_original_composite_image_2.png")
            #warp_tools.save_img(imatge_a_reg_path_1, img)
            #
            #imatge_a_reg = cv2.imread(imatge_a_reg_path)

            padded_img = transform.warp(img, img_obj.T, preserve_range=True,
                                        output_shape=img_obj.padded_shape_rc)
            #Codi meu
            #print(f'Mida padded_img: {img_obj.padded_shape_rc}')
            #[1537, 1537]

            composite_img_list[i] = padded_img

            img_corners_rc = warp_tools.get_corners_of_image(img.shape[0:2])
            warped_corners_xy = warp_tools.warp_xy(img_corners_rc[:, ::-1], img_obj.T)
            #Codi meu
            #print(f'Mida que se li passa al warp_tools.warp_xy: img_corners_rc[:, ::-1]: {img_corners_rc[:, ::-1]}')
            #[[   0    0]
            # [1063    0]
            # [1063 1149]
            # [   0 1149]]
            #
            min_r = min(warped_corners_xy[:, 1].min(), min_r)
            max_r = max(warped_corners_xy[:, 1].max(), max_r)
            min_c = min(warped_corners_xy[:, 0].min(), min_c)
            max_c = max(warped_corners_xy[:, 0].max(), max_c)
            #Codi meu
            #print(f'Mides finals: min_r: {min_r}, max_r: {max_r}, min_c: {min_c}, max_c: {max_c}')
            #


        composite_img = np.dstack(composite_img_list)
        cmap = viz.jzazbz_cmap()
        channel_colors = viz.get_n_colors(cmap, composite_img.shape[2])
        overlap_img = viz.color_multichannel(composite_img, channel_colors,
                                             rescale_channels=True,
                                             normalize_by="channel",
                                             cspace="CAM16UCS")

        min_r = int(min_r)
        max_r = int(np.ceil(max_r))
        min_c = int(min_c)
        max_c = int(np.ceil(max_c))
        #Codi meu
        #print(f'Mides finals: min_r: {min_r}, max_r: {max_r}, min_c: {min_c}, max_c: {max_c}')
        #
        overlap_img = overlap_img[min_r:max_r, min_c:max_c]
        overlap_img = (255*overlap_img).astype(np.uint8)

        return overlap_img

    def measure_original_mmi(self, img1, img2):
        """Measure Mattes mutation inormation between 2 unregistered images.
        """

        dst_rc = np.max([img1.shape, img2.shape], axis=1)
        padded_img_list = [None] * self.size
        for i, img in enumerate([img1, img2]):
            T = warp_tools.get_padding_matrix(img.shape, dst_rc)
            padded_img = transform.warp(img, T, preserve_range=True, output_shape=dst_rc)
            padded_img_list[i] = padded_img

        og_mmi = warp_tools.mattes_mi(padded_img_list[0], padded_img_list[1])

        return og_mmi

    def create_img_processor_dict(self, brightfield_processing_cls=DEFAULT_BRIGHTFIELD_CLASS,
                                  brightfield_processing_kwargs=DEFAULT_BRIGHTFIELD_PROCESSING_ARGS,
                                  if_processing_cls=DEFAULT_FLOURESCENCE_CLASS,
                                  if_processing_kwargs=DEFAULT_FLOURESCENCE_PROCESSING_ARGS,
                                  processor_dict=None):
        """Create dictionary to get processors for each image

        Create dictionary to get processors for each image. If an image is not in `processing_dict`,
        this function will try to guess the modality and then assign a default processor.

        Parameters
        ----------
        brightfield_processing_cls : ImageProcesser
            ImageProcesser to pre-process brightfield images to make them look as similar as possible.
            Should return a single channel uint8 image.

        brightfield_processing_kwargs : dict
            Dictionary of keyward arguments to be passed to `brightfield_processing_cls`

        if_processing_cls : ImageProcesser
            ImageProcesser to pre-process immunofluorescent images to make them look as similar as possible.
            Should return a single channel uint8 image.

        if_processing_kwargs : dict
            Dictionary of keyward arguments to be passed to `if_processing_cls`

        processor_dict : dict
            Each key should be the filename of the image, and the value either a subclassed
            preprocessing.ImageProcessor, or a list, where the 1st element is the processor,
            and the second element a dictionary of keyword arguments passed to ImageProcesser.process_img.
            If `None`, then this function will assign a processor to each image.

        Returns
        -------
        named_processing_dict : Dict
            Each key is the name of the slide, and the value is a list, where
            the 1st element is the processor, and the second element a dictionary
            of keyword arguments passed to ImageProcesser.process_img

        """

        if processor_dict is None:
            named_processing_dict = {}
        else:
            named_processing_dict = {self.get_slide(f).name: processor_dict[f] for f in processor_dict.keys()}

        for i, slide_obj in enumerate(self.slide_dict.values()):

            if slide_obj.name in named_processing_dict:
                slide_p = named_processing_dict[slide_obj.name]
                if isinstance(slide_p, list):
                    if len(slide_p) == 2:
                        slide_p, slide_kwargs = slide_p
                    elif len(slide_p) == 1:
                        # Provided processor, but no kwargs
                        slide_kwargs = {}
                else:
                    # Provided processor, but no kwargs
                    slide_kwargs = {}

                named_processing_dict[slide_obj.name] = [slide_p, slide_kwargs]

            else:
                # Processor not provided, so assign one based on inferred modality
                is_ihc = slide_obj.img_type == slide_tools.IHC_NAME
                if is_ihc:
                    processing_cls = brightfield_processing_cls
                    processing_kwargs = brightfield_processing_kwargs

                else:
                    processing_cls = if_processing_cls
                    processing_kwargs = if_processing_kwargs

                named_processing_dict[slide_obj.name] = [processing_cls, processing_kwargs]

        return named_processing_dict


    #Codi meu
    def generate_tissue_masks(self, slide_obj):
        """
        Genera màscares de teixit a partir de `rigid_reg_mask` de `slide_obj`.
        
        La primera vegada que es crida, genera la màscara de la meitat esquerra (`Màscara_teixit_HE`).
        La segona vegada que es crida, genera la màscara de la meitat dreta (`Màscara_teixit_IHC`).
        
        :param self: Instància de la classe (ha de contenir `self.mask_dir`).
        :param slide_obj: Objecte de la diapositiva per la qual es generaran les màscares.
        """
        # Comprovar si l'objecte de la imatge té `rigid_reg_mask`
        if not hasattr(slide_obj, 'rigid_reg_mask') or slide_obj.rigid_reg_mask is None:
            print(f"⚠️ ERROR: No s'ha trobat `rigid_reg_mask` per {slide_obj.name}.")
            return

        # Obtenir la mida de `rigid_reg_mask`
        #mask_height, mask_width = slide_obj.rigid_reg_mask.shape

        # Dividir en dues meitats
        #half_width = mask_width // 2
        #tissue_mask_he = slide_obj.rigid_reg_mask[:, :half_width]  # Meitat esquerra
        #tissue_mask_ihc = slide_obj.rigid_reg_mask[:, half_width:]  # Meitat dreta

        # Assegurar que el directori de màscares existeix
        os.makedirs(self.mask_dir_rigid_registration, exist_ok=True)

        # Comptador per controlar les crides a la funció
        if not hasattr(self, 'mask_call_count'):
            self.mask_call_count = 0
        self.mask_call_count += 1

        if self.mask_call_count == 1:
            # Guardar la màscara HE
            mask_he_path = os.path.join(self.mask_dir_rigid_registration, "Màscara_teixit_HE_original.png")
            warp_tools.save_img(mask_he_path, slide_obj.rigid_reg_mask)
            #mask_he_path = os.path.join(self.mask_dir_rigid_registration, "Màscara_teixit_HE.png")
            #cv2.imwrite(mask_he_path, tissue_mask_he)
            #print(f"✅ Màscara HE guardada a: {mask_he_path}")
        elif self.mask_call_count == 2:
            # Guardar la màscara IHC
            mask_ihc_path = os.path.join(self.mask_dir_rigid_registration, "Màscara_teixit_IHC_original.png")
            warp_tools.save_img(mask_ihc_path, slide_obj.rigid_reg_mask)
            #mask_ihc_path = os.path.join(self.mask_dir_rigid_registration, "Màscara_teixit_IHC.png")
            #cv2.imwrite(mask_ihc_path, tissue_mask_ihc)
            #print(f"✅ Màscara IHC guardada a: {mask_ihc_path}")
        else:
            print("⚠️ ERROR: Aquesta funció només es pot cridar dues vegades per cada execució.")
    #

    # Diccionari global per emmagatzemar la primera màscara en espera de la segona


    #Codi meu
    def compute_iou_from_masks(self, rigid_or_non_rigid):
        """
        Calcula l'error d'Intersecció sobre Unió (IoU) entre dues màscares en un directori donat **sense binaritzar-les**.

        - Es busquen imatges `.png` al directori especificat i es calcula l'IoU entre les dues primeres trobades.
        - Es genera la multiplicació de matrius píxel a píxel.
        - Es genera una imatge OR on si un dels dos píxels és blanc, el resultat serà blanc.

        :param rigid_or_non_rigid: 0 per registre rígid, 1 per no rígid.
        :return: Valor de l'IoU entre les dues màscares.
        """
        # Selecciona el directori correcte

        if rigid_or_non_rigid == 0:
            #Obtenir les imatges .png del directori
            mask_dir = self.mask_dir_rigid_registration
            mask_files = ["Màscara_teixit_HE_retallada.png", "Màscara_teixit_IHC_original.png"]
            mask_bbox = "_non_rigid_bbox_retallat.png"

            if len(mask_files) < 2:
                print("ERROR: No hi ha prou màscares per calcular IoU.")
                return None

            # Llegeix les dues màscares en escala de grisos
            mask1_path = os.path.join(mask_dir, mask_files[0])
            mask2_path = os.path.join(mask_dir, mask_files[1])
            mask_bbox_path = os.path.join(mask_dir, mask_bbox)

            mask1 = cv2.imread(mask1_path, cv2.IMREAD_GRAYSCALE)
            mask2 = cv2.imread(mask2_path, cv2.IMREAD_GRAYSCALE)
            mask_bbox = cv2.imread(mask_bbox_path, cv2.IMREAD_GRAYSCALE)

            if mask1 is None or mask2 is None:
                print("ERROR: No s'han pogut carregar les màscares.")
                return None

            #Multipliquem les dos màscares de teixit (Màscara HE i màscara IHC per la màscara on hi ha la bbox del teixit per evitar falsos teixits)
            mask1 = np.logical_and(mask1 > 0, mask_bbox > 0).astype(np.uint8) * 255
            mask2 = np.logical_and(mask2 > 0, mask_bbox > 0).astype(np.uint8) * 255 
            cv2.imwrite(os.path.join(mask_dir, "_mask_1_IoU.png"), mask1)
            cv2.imwrite(os.path.join(mask_dir, "_mask_2_IoU.png"), mask2)        

        else:
            mask_files = ["0_Slide_mask_6_non_registered.png", "1_Slide_mask_6.png"]
            mask_dir = self.mask_dir_non_rigid_registration

            if len(mask_files) < 2:
                print("ERROR: No hi ha prou màscares per calcular IoU.")
                return None

            # Llegeix les dues màscares en escala de grisos
            mask1_path = os.path.join(mask_dir, mask_files[0])
            mask2_path = os.path.join(mask_dir, mask_files[1])

            mask1 = cv2.imread(mask1_path, cv2.IMREAD_GRAYSCALE)
            mask2 = cv2.imread(mask2_path, cv2.IMREAD_GRAYSCALE)

            if mask1 is None or mask2 is None:
                print("ERROR: No s'han pogut carregar les màscares.")
                return None


        # Calcular intersecció i unió directament sense binaritzar
        intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
        union = np.logical_or(mask1 > 0, mask2 > 0).sum()

        if union == 0: #Per evitar dividir per 0
            return 0.0
        
        elif intersection == 0: #El registre ha sigut tant dolent que no hi ha intersecció, per tant IoU = 0
            return 0.0

        iou = intersection / union
        print(f"IoU entre {mask_files[0]} i {mask_files[1]}: {iou:.4f}")
        print(f"Intersecció: {intersection:.4f}")
        print(f"Union: {union:.4f}")


        # Multiplicació i OR
        mask1 = mask1.astype(np.uint8)
        mask2 = mask2.astype(np.uint8)
        mask_multiplication = cv2.bitwise_and(mask1, mask2)
        mask_or = cv2.bitwise_or(mask1, mask2)
        if rigid_or_non_rigid == 0:
            cv2.imwrite(os.path.join(mask_dir, "mask_multiplication_rigid.png"), mask_multiplication)
            cv2.imwrite(os.path.join(mask_dir, "mask_or_rigid.png"), mask_or)
        else:
            cv2.imwrite(os.path.join(mask_dir, "mask_multiplication_non_rigid.png"), mask_multiplication)
            cv2.imwrite(os.path.join(mask_dir, "mask_or_non_rigid.png"), mask_or)

        return iou

    #


    #Codi meu
    def histeq_mask(self, image, mask):
        """
        Equalitza una imatge només dins la màscara (pixels on mask == 1).

        :param image: ndarray, imatge en escala de grisos (float o uint8)
        :param mask: ndarray, màscara binària (bool o 0/1)
        :return: imatge equalitzada amb els valors fora màscara iguals als originals
        """
        if image.shape != mask.shape:
            raise ValueError("La imatge i la màscara han de tenir la mateixa mida.")

        image = img_as_float(image)
        mask = mask.astype(bool)

        values = image[mask]  # només valors dins la màscara
        values_eq = exposure.equalize_hist(values)  # equalització només dins màscara

        image_eq = np.copy(image)
        image_eq[mask] = values_eq
        return image_eq

    #Codi meu
    def compute_corr_from_masks(self, rigid_or_non_rigid):
        """
        Carrega imatges i màscares, equalitza dins les màscares, i calcula la correlació normalitzada.

        :param he_img_path: Ruta a la imatge HE (color o escala de grisos)
        :param he_mask_path: Ruta a la màscara HE (blanc = teixit)
        :param ihc_img_path: Ruta a la imatge IHC
        :param ihc_mask_path: Ruta a la màscara IHC
        :return: correlació entre les imatges dins la intersecció de màscares
        """

        if rigid_or_non_rigid == 0:
            #Obtenir les imatges .png del directori
            mask_dir = self.mask_dir_rigid_registration

            he_img_path = os.path.join(mask_dir, "Imatge_teixit_HE_retallada_rigid_registered.png")
            ihc_img_path = os.path.join(mask_dir, "Imatge_teixit_IHC.png")
            he_mask_path = os.path.join(mask_dir, "Màscara_teixit_HE_retallada.png")
            ihc_mask_path = os.path.join(mask_dir, "Màscara_teixit_IHC_original.png")
            mask_bbox_path = os.path.join(mask_dir, "_non_rigid_bbox_retallat.png")

            mask1 = imread(he_img_path, as_gray=True)
            mask2 = imread(ihc_img_path, as_gray=True)
            mask3 = imread(he_mask_path, as_gray=True)
            mask4 = imread(ihc_mask_path, as_gray=True)
            mask_bbox = imread(mask_bbox_path, as_gray=True)

            #Multipliquem les imatges per la bbox
            mask1_1 = mask1 * mask_bbox
            mask2_1 = mask2 * mask_bbox
            mask3_1 = mask3 * mask_bbox
            mask4_1 = mask4 * mask_bbox

            he_img_path = os.path.join(mask_dir, "Imatge_teixit_HE_retallada_rigid_registered_1.png")
            warp_tools.save_img(he_img_path, mask1_1)
            ihc_img_path = os.path.join(mask_dir, "Imatge_teixit_IHC_1.png")
            warp_tools.save_img(ihc_img_path, mask2_1)
            he_mask_path = os.path.join(mask_dir, "Màscara_teixit_HE_retallada_1.png")
            warp_tools.save_img(he_mask_path, mask3_1)
            ihc_mask_path = os.path.join(mask_dir, "Màscara_teixit_IHC_original_1.png")
            warp_tools.save_img(ihc_mask_path, mask4_1)            


        elif rigid_or_non_rigid == 1:
            #Obtenir les imatges .png del directori
            mask_dir = self.mask_dir_non_rigid_registration

            he_img_path = os.path.join(mask_dir, "0_Slide_mask_4_non_registered.png")
            ihc_img_path = os.path.join(mask_dir, "1_Slide_mask_4.png")
            he_mask_path = os.path.join(mask_dir, "0_Slide_mask_6_non_registered.png")
            ihc_mask_path = os.path.join(mask_dir, "1_Slide_mask_6.png")

        # Carrega les imatges (amb as_gray=True → retorna float [0,1])
        he_img = imread(he_img_path, as_gray=True)
        ihc_img = imread(ihc_img_path, as_gray=True)


        # Carrega les màscares i les binaritza
        he_mask = cv2.imread(he_mask_path, cv2.IMREAD_GRAYSCALE) > 0
        ihc_mask = cv2.imread(ihc_mask_path, cv2.IMREAD_GRAYSCALE) > 0
        warp_tools.save_img(os.path.join(mask_dir, "Màscara_teixit_HE_original_3.png"), he_mask)
        warp_tools.save_img(os.path.join(mask_dir, "Màscara_teixit_IHC_original_2.png"), ihc_mask)

        #Plots dels histogrames
        #plt.hist(he_img.ravel(), bins=256, range=(0, 256), color='gray')
        #plt.title("Histograma HE - Escala de grisos")
        #plt.xlabel("Intensitat")
        #plt.ylabel("Nombre de píxels")
        #plt.show()

        #plt.hist(ihc_img.ravel(), bins=256, range=(0, 256), color='gray')
        #plt.title("Histograma IHC - Escala de grisos")
        #plt.xlabel("Intensitat")
        #plt.ylabel("Nombre de píxels")
        #plt.show()

        # Equalitza les imatges dins les seves màscares
        he_eq = self.histeq_mask(he_img, he_mask)
        ihc_eq = self.histeq_mask(ihc_img, ihc_mask)

        # Intersecció de màscares
        intersection_mask = he_mask & ihc_mask

        #Accedeix a la he_eq que és la imatge HE equalitzada només en la seva zona de teixit, i s'extreu aquells valors de la 
        #HE equalitzada que estiguin en la intersecció de les dos màscares HE i IHC
        vals1 = he_eq[intersection_mask] #Vector 1D amb els píxels d'HE equalitzats que formen part de la intersecció de les dos màscares HE i IHC
        vals2 = ihc_eq[intersection_mask]

        #res1_path = os.path.join(mask_dir, "res1.png")
        #warp_tools.save_img(res1_path, vals1)
        #res2_path = os.path.join(mask_dir, "res2.png")
        #warp_tools.save_img(res2_path, vals2)
        #res3_path = os.path.join(mask_dir, "res3.png")
        #warp_tools.save_img(res3_path, intersection_mask)

        if len(vals1) == 0:
            correlation= 0
            raise ValueError("La intersecció de màscares és buida.")

        # Correlació normalitzada
        numerator = np.sum(vals1 * vals2)
        E_HE = np.sqrt(np.sum(vals1 ** 2))
        E_IHC = np.sqrt(np.sum(vals2 ** 2))
        denominator = E_HE * E_IHC

        if denominator == 0:
            return 0.0

        correlation = numerator / denominator
        print(f"Correlation: {correlation:.4f}")
        print(f"E_HE: {E_HE:.4f}")
        print(f"E_IHC: {E_IHC:.4f}")
        return correlation


    #Codi meu
    def eliminar_padding_calculat(self, path_HE_original, level):
        """
        Calcula i elimina el padding d'una imatge delmada .png, després d'escalar-la al nivell de piràmide indicat.
        
        Parameters:
            path_HE_original (str): Ruta a la imatge .mrxs original.
            path_imatge_delmada (str): Ruta a la imatge .png amb padding (normalment a nivell baix).
            level (int): Nivell de piràmide a treballar (0 = sense delmatge, 8 = molt delmat).
        
        Returns:
            img_sense_padding (PIL.Image): Imatge a nivell `level` sense padding.
        """

        # 1. Obtenir la mida original (nivell 0)
        print(f"Ruta a la img_HE_original: {path_HE_original}")
        slide = openslide.OpenSlide(str(path_HE_original))
        mida_mrxs_lvl0 = slide.dimensions  # (w, h)
        print("Mida original (nivell 0):", mida_mrxs_lvl0)

        # Obtenir la mida real de la imatge útil (Les dimensions que dona QuPath)
        bounds_width = int(slide.properties[openslide.PROPERTY_NAME_BOUNDS_WIDTH])
        bounds_height = int(slide.properties[openslide.PROPERTY_NAME_BOUNDS_HEIGHT])
        print("Mida útil (com la mostra QuPath):", (bounds_width, bounds_height))

        # 2. Carregar imatge .png amb padding (en un nivell desconegut)
        img_etiquetes_he = Image.open(self.path_img_etiquetes_he).convert('L')
        img_np = np.array(img_etiquetes_he)
        h_png, w_png = img_np.shape
        print("Mida de la imatge PNG amb padding:", (w_png, h_png))

        # 3. Determinar el millor factor de delmatge (nivell més proper)
        w0, h0 = bounds_width, bounds_height
        factors = [2**i for i in range(10)]
        errors = [abs((w0 // f) - w_png) + abs((h0 // f) - h_png) for f in factors]
        idx_min = np.argmin(errors)
        millor_factor = factors[idx_min]
        print(f"Millor factor trobat: {millor_factor} (nivell {idx_min})")

        # 4. Mida esperada sense padding al nivell detectat
        w_esperat = w0 // millor_factor
        h_esperat = h0 // millor_factor

        # 5. Padding a nivell detectat
        padding_w = w_png - w_esperat
        padding_h = h_png - h_esperat
        print(f"Padding total (a nivell {idx_min}): (h={padding_h}, w={padding_w})")

        # 6. Delmar la imatge segons el nivell objectiu (`level`)
        escala = 2 ** (level - idx_min)
        if escala != 1:
            img_np = np.array(img_etiquetes_he.resize(
                (w_png // escala, h_png // escala),
                resample=Image.NEAREST
            ))
            h_delmat, w_delmat = img_np.shape
        else:
            h_delmat, w_delmat = h_png, w_png
            img_np = np.array(img_etiquetes_he)

        # 7. Delma també el padding al nivell actual
        padding_w_nivell = round(padding_w / (escala))
        padding_h_nivell = round(padding_h / (escala))
        print(f"Level = {escala}")
        print(f"Padding a eliminar a nivell {level}: (h={padding_h_nivell}, w={padding_w_nivell})")

        # 8. Retallar la imatge delmada per eliminar el padding
        nova_array = img_np[:h_delmat - padding_h_nivell, :w_delmat - padding_w_nivell]
        img_sense_padding = Image.fromarray(nova_array)
        print(f"[✔] Imatge retallada: nova mida {(nova_array.shape[1], nova_array.shape[0])}")

        path_save = os.path.join(self.mask_dir_preprocessing, "HE_etiquetes_sense_padding.png")
        warp_tools.save_img(path_save, img_sense_padding)

        return img_sense_padding


    #Codi meu
    def preparar_imatge_HE_etiquetes(self, path_HE_original, imatge_HE_etiquetes_sense_padding, level, rigid_or_non_rigid):

        img_etiquetes_he = imatge_HE_etiquetes_sense_padding
        # Taula de dimensions de cada nivell (w, h)
        dimensions_per_nivell = np.array([
            [272128, 294144],
            [136064, 147072],
            [ 68032,  73536],
            [ 34016,  36768],
            [ 17008,  18384],
            [  8504,   9192],
            [  4252,   4596],
            [  2126,   2298],
            [  1063,   1149],
            [   531,    574]
        ])

        # Obre la slide i obté el padding original
        slide = openslide.OpenSlide(str(path_HE_original))
        offset_x = int(slide.properties.get(openslide.PROPERTY_NAME_BOUNDS_X, 0))
        offset_y = int(slide.properties.get(openslide.PROPERTY_NAME_BOUNDS_Y, 0))
        print(f"Offset nivell 0 (pixels): ({offset_x}, {offset_y})")

        # Escala l'offset segons el nivell i arrodoneix
        scale_factor = 2 ** level
        offset_x_lvl = round(offset_x / scale_factor)
        offset_y_lvl = round(offset_y / scale_factor)
        print(f"Offset nivell {level} arrodonit: ({offset_x_lvl}, {offset_y_lvl})")

        # Obté les dimensions del canvas per al nivell
        canvas_w, canvas_h = dimensions_per_nivell[level]
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)  # Format (h, w)

        # Converteix la imatge d'etiquetes a array
        img_np = np.array(img_etiquetes_he)
        h, w = img_np.shape

        # Insereix la imatge al canvas a la posició calculada
        canvas[offset_y_lvl:offset_y_lvl + h, offset_x_lvl:offset_x_lvl + w] = img_np

        # Converteix a imatge PIL i desa-la com PNG
        if rigid_or_non_rigid == 0:
            img_final = Image.fromarray(canvas)
            self.img_etiquetes_he = img_final
            path = os.path.join(self.mask_dir_preprocessing, "HE_etiquetes_Valis.png")
            path2 = os.path.join(self.mask_dir_rigid_registration, "HE_etiquetes_Valis.png")
            warp_tools.save_img(path, img_final)
            warp_tools.save_img(path2, img_final)
        elif rigid_or_non_rigid == 1:
            img_final = Image.fromarray(canvas)
            path = os.path.join(self.mask_dir_non_rigid_registration, "HE_etiquetes_Valis_non_register.png")
            warp_tools.save_img(path, img_final)
    #


    def preparar_imatge_HE_etiquetes_inversa(self, path_IHC_original, imatge_HE_etiquetes_non_registered):

        img_etiquetes_he = imatge_HE_etiquetes_non_registered
        # Taula de dimensions de cada nivell (w, h)
        dimensions_per_nivell = np.array([
            [272128, 294144],
            [136064, 147072],
            [ 68032,  73536],
            [ 34016,  36768],
            [ 17008,  18384],
            [  8504,   9192],
            [  4252,   4596],
            [  2126,   2298],
            [  1063,   1149],
            [   531,    574]
        ])

        # Dimensions de la primera imatge
        h1, w1 = img_etiquetes_he.shape[:2]

        # Obre la slide .mrxs
        slide = openslide.OpenSlide(path_IHC_original)

        # Troba el nivell més alt on la mida sigui menor o igual a la imatge
        nivell = None
        for i, (w, h) in enumerate(dimensions_per_nivell):
            if w <= w1 and h <= h1:
                nivell = i
                break
        if nivell is None:
            raise ValueError("Cap nivell té dimensions més petites que la imatge original.")


        dimensions = slide.dimensions  # (amplada, alçada)
        #print("Dimensions IHC", dimensions)
        # Obté el padding
        offset_x = int(slide.properties.get(openslide.PROPERTY_NAME_BOUNDS_X, 0))
        offset_y = int(slide.properties.get(openslide.PROPERTY_NAME_BOUNDS_Y, 0))
        print(f"Offset nivell 0 (pixels): ({offset_x}, {offset_y})")
        scale_factor = 2 ** nivell
        offset_x_lvl = round(offset_x / scale_factor)
        offset_y_lvl = round(offset_y / scale_factor)
        print(f"Offset nivell {nivell} arrodonit: ({offset_x_lvl}, {offset_y_lvl})")

        # Dimensions originals que diu el qupath
        bounds_width = int(slide.properties.get(openslide.PROPERTY_NAME_BOUNDS_WIDTH, slide.dimensions[0]))
        bounds_height = int(slide.properties.get(openslide.PROPERTY_NAME_BOUNDS_HEIGHT, slide.dimensions[1]))
        print(f"Dimensions de la imatge IHC al nivell 0: ({bounds_width}, {bounds_height})")


        #Calculem el offset que ha de tenir per passar de la piràmide de IHC a la piramide de Valis
        dim_IHC_nivell_x = round(bounds_width / scale_factor)
        dim_IHC_nivell_y = round(bounds_height / scale_factor)
        print(f"Dimensions que hauria de tenir la imatge IHC al nivell {nivell}: ({dim_IHC_nivell_x}, {dim_IHC_nivell_y})")


        # Retalla i centra la imatge
        img_np = np.array(img_etiquetes_he)
        img_np_centrada = img_np[offset_y_lvl:offset_y_lvl + dim_IHC_nivell_y, offset_x_lvl:offset_x_lvl + dim_IHC_nivell_x]

        img_retallada = Image.fromarray(img_np_centrada)
        path = os.path.join(self.mask_dir_non_rigid_registration, "HE_etiquetes_Valis_non_registered_mida_IHC.png")
        warp_tools.save_img(path, img_retallada)
        
    #

    def process_imgs(self, processor_dict):
        """Process images to make them look as similar as possible

        Images will also be normalized after images are processed

        Parameters
        ----------
        processor_dict : dict
            Each key should be the filename of the image, and the value either a subclassed
            preprocessing.ImageProcessor, or a list, where the 1st element is the processor,
            and the second element a dictionary of keyword arguments passed to the processor.
            If `None`, then a default processor will be used for each image based on
            the inferred modality.

        """

        pathlib.Path(self.processed_dir).mkdir(exist_ok=True, parents=True)
        if self.norm_method is not None:
            if self.norm_method == "histo_match":
                ref_histogram = np.zeros(256, dtype=np.int)
            else:
                all_v = [None]*self.size

        for i, slide_obj in enumerate(tqdm.tqdm(self.slide_dict.values(), desc=PROCESS_IMG_MSG, unit="image")):


            #Codi meu
            #print(f"Mida de la màscara abans d'entrar al preprocessing: {slide_obj.slide_dimensions_wh}")
            
            #Vull guardar la màscara de teixit amb una altra dimensió
            
            #processing_cls, processing_kwargs = processor_dict[slide_obj.name]
            """
            processor = processing_cls(image=slide_obj.image,
                                       src_f=slide_obj.src_f,
                                       level=4,
                                       series=slide_obj.series,
                                       reader=slide_obj.reader)
            """ #Obs tenir en compte que el level = 0 és el nivell amb més pixels de la piràmide, per tant el quin té més resolució i el nivell 9 el quin menys
            #processed_img = processor.process_image()
            if i == 0:
                warp_tools.save_img(os.path.join(self.mask_dir_rigid_serial_rigid, "Imatge_teixit_HE_retallada.png"),  slide_obj.image)
            elif i == 1:
                warp_tools.save_img(os.path.join(self.mask_dir_rigid_registration, "Imatge_teixit_IHC.png"),  slide_obj.image)
            #

            levels_in_range = np.where(slide_obj.slide_dimensions_wh.max(axis=1) < self.max_processed_image_dim_px)[0]
            if len(levels_in_range) > 0:
                level = levels_in_range[0]
            else:
                level = len(slide_obj.slide_dimensions_wh) - 1
            #Codi meu
            #print(f"Mida del level: {level}")
            #

            processing_cls, processing_kwargs = processor_dict[slide_obj.name]
            processor = processing_cls(image=slide_obj.image,
                                       src_f=slide_obj.src_f,
                                       level=level,
                                       series=slide_obj.series,
                                       reader=slide_obj.reader)

            try:
                processed_img = processor.process_image(**processing_kwargs)

                #Codi meu
                #print(f"Mida de la màscara que retorna preprocessing: {processed_img.shape}")
                #

            except TypeError:
                # processor.process_image doesn't take kwargs
                processed_img = processor.process_image()

            #Codi meu
                #print(f"Mida de la màscara que retorna preprocessing: {processed_img.shape}")

            #warp_tools.save_img(os.path.join(self.mask_dir, f'{slide_obj.name}_3.png'),  processed_img)
            #

            processed_img = exposure.rescale_intensity(processed_img, out_range=(0, 255)).astype(np.uint8)
            scaling = np.min(self.max_processed_image_dim_px/np.array(processed_img.shape[0:2]))
            
            if scaling < 1:
                processed_img = warp_tools.rescale_img(processed_img, scaling)

            if self.create_masks:
                # Get masks #
                pathlib.Path(self.mask_dir).mkdir(exist_ok=True, parents=True)

                # Slice region from slide and process too
                mask = processor.create_mask()
                if not np.all(mask.shape == processed_img.shape[0:2]):
                    mask = warp_tools.resize_img(mask, processed_img.shape[0:2], interp_method="nearest")

                slide_obj.rigid_reg_mask = mask

                #Codi meu
                #warp_tools.save_img(os.path.join(self.mask_dir, f'{slide_obj.name}_1.png'),  mask)
                #print(f"Màscara de teixit amb mida original")
                #print(f"Mida de `mask` per {slide_obj.name}: {mask.shape}")
                #


                # Save image with mask drawn on top of it
                thumbnail_mask = self.create_thumbnail(mask)
                if slide_obj.img_type == slide_tools.IHC_NAME:
                    thumbnail_img = self.create_thumbnail(slide_obj.image)
                else:
                    thumbnail_img = self.create_thumbnail(processed_img)

                thumbnail_mask_outline = viz.draw_outline(thumbnail_img, thumbnail_mask)
                outline_f_out = os.path.join(self.mask_dir_preprocessing, f'{slide_obj.name}.png')
                warp_tools.save_img(outline_f_out, thumbnail_mask_outline)

            else:
                mask = np.full(processed_img.shape, 255, dtype=np.uint8)

            slide_obj.rigid_reg_mask = mask
            slide_obj.processed_img = processed_img
            

            processed_f_out = os.path.join(self.processed_dir, slide_obj.name + ".png")
            slide_obj.processed_img_f = processed_f_out
            slide_obj.processed_img_shape_rc = np.array(processed_img.shape[0:2])
            warp_tools.save_img(processed_f_out, processed_img)

            img_for_stats = processed_img.reshape(-1)

            if self.norm_method is not None:
                if self.norm_method == "histo_match":
                    img_hist, _ = np.histogram(img_for_stats, bins=256)
                    ref_histogram += img_hist
                else:
                    all_v[i] = img_for_stats.reshape(-1)
            
            #Codi meu
            self.generate_tissue_masks(slide_obj)
            #

        if self.norm_method is not None:
            if self.norm_method == "histo_match":
                target_stats = ref_histogram
            else:
                all_v = np.hstack(all_v)
                target_stats = all_v

            self.normalize_images(target_stats)



        

    def denoise_images(self):
        for i, slide_obj in enumerate(tqdm.tqdm(self.slide_dict.values(), desc=DENOISE_MSG, unit="image")):
            if slide_obj.rigid_reg_mask is None:
                is_ihc = slide_obj.img_type == slide_tools.IHC_NAME
                _, tissue_mask = preprocessing.create_tissue_mask(slide_obj.image, is_ihc)
                
                #Codi meu
                """
                # Crear el directori "masks/" si no existeix
                mask_dir = os.path.join(self.mask_dir, "masks")
                if not os.path.exists(mask_dir):
                    os.makedirs(mask_dir)

                # Definir la ruta on es guardarà la màscara
                mask_filename = f"{slide_obj.name}_original_mask.png"
                mask_path = os.path.join(mask_dir, mask_filename)

                # Guardar la màscara amb la seva mida original
                cv2.imwrite(mask_path, tissue_mask)

                print(f"Màscara de teixit guardada amb mida original a: {mask_path}")
                """
                #

                mask_bbox = warp_tools.xy2bbox(warp_tools.mask2xy(tissue_mask))
                c0, r0 = mask_bbox[:2]
                c1, r1 = mask_bbox[:2] + mask_bbox[2:]
                denoise_mask = np.zeros_like(tissue_mask)
                denoise_mask[r0:r1, c0:c1] = 255
            else:
                denoise_mask = slide_obj.rigid_reg_mask
                
                #Codi meu
                """
                # Crear el directori "masks/" si no existeix
                mask_dir = os.path.join(self.mask_dir, "masks")
                if not os.path.exists(mask_dir):
                    os.makedirs(mask_dir)

                #Codi meu
                # Definir la ruta on es guardarà la màscara
                mask_filename = f"{slide_obj.name}_original_mask.png"
                mask_path = os.path.join(mask_dir, mask_filename)

                # Guardar la màscara amb la seva mida original
                cv2.imwrite(mask_path, denoise_mask)

                print(f"Màscara de teixit guardada amb mida original a: {mask_path}")
                """
                #


            denoised = preprocessing.denoise_img(slide_obj.processed_img, mask=denoise_mask)
            warp_tools.save_img(slide_obj.processed_img_f, denoised)

    def normalize_images(self, target):
        """Normalize intensity values in images

        Parameters
        ----------
        target : ndarray
            Target statistics used to normalize images

        """
        # print("\n==== Normalizing images\n")
        for i, slide_obj in enumerate(tqdm.tqdm(self.slide_dict.values(), desc=NORM_IMG_MSG, unit="image")):
            vips_img = pyvips.Image.new_from_file(slide_obj.processed_img_f)
            img = warp_tools.vips2numpy(vips_img)
            if self.norm_method == "histo_match":
                self.target_processing_stats = target
                normed_img = preprocessing.match_histograms(img, self.target_processing_stats)
            elif self.norm_method == "img_stats":
                self.target_processing_stats = preprocessing.get_channel_stats(target)
                normed_img = preprocessing.norm_img_stats(img, self.target_processing_stats)

            normed_img = exposure.rescale_intensity(normed_img, out_range=(0, 255)).astype(np.uint8)
            slide_obj.processed_img = normed_img

            slide_obj.processed_img_shape_rc = np.array(normed_img.shape[0:2])
            warp_tools.save_img(slide_obj.processed_img_f, normed_img)

    def create_thumbnail(self, img, rescale_color=False):
        """Create thumbnail image to view results
        """

        is_vips = isinstance(img, pyvips.Image)

        img_shape = warp_tools.get_shape(img)
        scaling = np.min(self.thumbnail_size/np.array(img_shape[:2]))
        if scaling < 1:
            thumbnail = warp_tools.rescale_img(img, scaling)
        else:
            thumbnail = img

        if rescale_color is True:
            if is_vips:
                # Convert to numpy to rescale
                thumbnail = warp_tools.vips2numpy(img)
            thumbnail = exposure.rescale_intensity(thumbnail, out_range=(0, 255)).astype(np.uint8)

            if is_vips:
                # Convert back to pyvips
                thumbnail = warp_tools.numpy2vips(thumbnail)

        return thumbnail

    def draw_overlap_img(self, img_list):
        """Create image showing the overlap of registered images
        """

        composite_img = np.dstack(img_list)
        #Codi meu  Aqui la imatge ja és de mida [1673, 1649]
        path = os.path.join(self.mask_dir_rigid_registration, "Imatge_draw_overlap_img.png")
        warp_tools.save_img(path, composite_img)
        mida = composite_img.shape[:2]
        print(f'mida: {mida}')
        #

        cmap = viz.jzazbz_cmap()
        channel_colors = viz.get_n_colors(cmap, composite_img.shape[2])
        overlap_img = viz.color_multichannel(composite_img, channel_colors,
                                             rescale_channels=True,
                                             normalize_by="channel",
                                             cspace="CAM16UCS")

        overlap_img = exposure.equalize_adapthist(overlap_img)
        overlap_img = exposure.rescale_intensity(overlap_img, out_range=(0, 255)).astype(np.uint8)

        return overlap_img

    #Codi meu
    def draw_overlap_binary_masks(self, mask1, mask2):
        """
        Genera una imatge RGB de superposició de dues màscares binàries amb fons blanc.
        - mask1 → verd
        - mask2 → magenta
        - fons → blanc
        """
        # Assegura que són binàries
        mask1 = (mask1 > 0).astype(np.uint8)
        mask2 = (mask2 > 0).astype(np.uint8)

        # Crea imatge RGB en blanc
        h, w = mask1.shape
        overlap_img = np.ones((h, w, 3), dtype=np.uint8) * 255  # fons blanc

        # Pinta els teixits
        # Verd: [0, 255, 0], Magenta: [255, 0, 255]

        # Teixit 1 només → verd
        overlap_img[(mask1 == 1) & (mask2 == 0)] = [0, 255, 0]

        # Teixit 2 només → magenta
        overlap_img[(mask2 == 1) & (mask1 == 0)] = [255, 0, 255]

        # Teixit solapat → un altre color (gris, o un mix personalitzat)
        overlap_img[(mask1 == 1) & (mask2 == 1)] = [128, 128, 128]

        return overlap_img
    #

    def get_ref_img_mask(self, rigid_registrar):
        """Create mask that covers reference image

        Returns
        -------
        mask : ndarray
            Mask that covers reference image in registered images
        mask_bbox_xywh : tuple of int
            XYWH of mask in reference image

        """

        ref_name = self.name_dict[self.reference_img_f]
        ref_slide = rigid_registrar.img_obj_dict[ref_name]
        ref_shape_wh = ref_slide.image.shape[0:2][::-1]

        uw_mask = np.full(ref_shape_wh[::-1], 255, dtype=np.uint8)
        mask = warp_tools.warp_img(uw_mask, ref_slide.M,
                                   out_shape_rc=ref_slide.registered_shape_rc)

        reg_txy = -ref_slide.M[0:2, 2]
        mask_bbox_xywh = np.array([*reg_txy, *ref_shape_wh])
        
        #Codi meu
        prova = "_ref_img_mask"
        outline_f_out = os.path.join(self.mask_dir_rigid_registration, f'{ref_name}{prova}.png')
        warp_tools.save_img(outline_f_out, mask)
        print(f"Mida de la matriu ref_img_mask: {ref_shape_wh}")
        print(f"Desada a : {outline_f_out}")
        #


        return mask, mask_bbox_xywh

    def get_all_overlap_mask(self, rigid_registrar):
        """Create mask that covers all tissue


        Returns
        -------
        mask : ndarray
            Mask that covers reference image in registered images
        mask_bbox_xywh : tuple of int
            XYWH of mask in reference image

        """

        ref_name = self.name_dict[self.reference_img_f]
        ref_slide = rigid_registrar.img_obj_dict[ref_name]
        combo_mask = np.zeros(ref_slide.registered_shape_rc, dtype=int)
        for img_obj in rigid_registrar.img_obj_list:

            img_mask = self.slide_dict[img_obj.name].rigid_reg_mask
            warped_img_mask = warp_tools.warp_img(img_mask,
                                                  M=img_obj.M,
                                                  out_shape_rc=img_obj.registered_shape_rc,
                                                  interp_method="nearest")

            combo_mask[warped_img_mask > 0] += 1

        temp_mask = 255*filters.apply_hysteresis_threshold(combo_mask, 0.5, self.size-0.5).astype(np.uint8)
        mask = 255*ndimage.binary_fill_holes(temp_mask).astype(np.uint8)
        mask = preprocessing.mask2contours(mask)

        mask_bbox_xywh = warp_tools.xy2bbox(warp_tools.mask2xy(mask))
        
        #Codi meu
        prova = "_all_overlap_mask"
        outline_f_out = os.path.join(self.mask_dir_rigid_registration, f'{ref_name}{prova}.png')
        warp_tools.save_img(outline_f_out, mask)
        print(f"Mida de la matriu all_overlap_mask: {mask.shape}")
        print(f"Desada a : {outline_f_out}")
        #

        return mask, mask_bbox_xywh



    def get_null_overlap_mask(self, rigid_registrar):
        """Create mask that covers all of the image.
        Not really a mask


        Returns
        -------
        mask : ndarray
            Mask that covers reference image in registered images
        mask_bbox_xywh : tuple of int
            XYWH of mask in reference image

        """
        reg_shape = rigid_registrar.img_obj_list[0].registered_shape_rc
        #Codi meu
        #
        #
        mask = np.full(reg_shape, 255, dtype=np.uint8)
        mask_bbox_xywh = np.array([0, 0, reg_shape[1], reg_shape[0]])

        #Codi meu
        ref_name = self.name_dict[self.reference_img_f]
        prova = "_null_overlap_mask"
        outline_f_out = os.path.join(self.mask_dir_rigid_registration, f'{ref_name}{prova}.png')
        warp_tools.save_img(outline_f_out, mask)
        print(f"Mida de la matriu null_overlap_mask: {reg_shape}")
        print(f"Desada a : {outline_f_out}")
        #

        return mask, mask_bbox_xywh

    def create_crop_masks(self, rigid_registrar):
        """Create masks based on rigid registration

        """
        mask_dict = {}
        mask_dict[CROP_REF] = self.get_ref_img_mask(rigid_registrar)
        mask_dict[CROP_OVERLAP] = self.get_all_overlap_mask(rigid_registrar)
        mask_dict[CROP_NONE] = self.get_null_overlap_mask(rigid_registrar)
        self.mask_dict = mask_dict

    def get_crop_mask(self, overlap_type):
        """Get overlap mask and bounding box

        Returns
        -------
        mask : ndarray
            Mask

        mask_xywh : tuple
            XYWH for bounding box around mask

        """
        if overlap_type is None:
            overlap_type = CROP_NONE

        return self.mask_dict[overlap_type]
        

    def rigid_register_partial(self, tform_dict=None):
        """Perform rigid registration using provided parameters

        Still sorts images by similarity for use with non-rigid registration.

        tform_dict : dictionary
            Dictionary with rigid registration parameters. Each key is the image's file name, and
            the values are another dictionary with transformation parameters:
                M: 3x3 inverse transformation matrix. Found by determining how to align fixed to moving.
                    If M was found by determining how to align moving to fixed, then it will need to be inverted

                transformation_src_shape_rc: shape (row, col) of image used to find the rigid transformation. If
                    not provided, then it is assumed to be the shape of the level 0 slide
                transformation_dst_shape_rc: shape of registered image. If not presesnt, but a reference was provided
                    and `transformation_src_shape_rc` was not provided, this is assumed to be the shape of the reference slide

            If None, then all rigid M will be the identity matrix
        """


        # Still need to sort images #
        rigid_registrar = serial_rigid.SerialRigidRegistrar(self.processed_dir,
                                        imgs_ordered=self.imgs_ordered,
                                        reference_img_f=self.reference_img_f,
                                        name=self.name,
                                        align_to_reference=self.align_to_reference)

        feature_detector = self.rigid_reg_kwargs[FD_KEY]
        matcher = self.rigid_reg_kwargs[MATCHER_KEY]
        similarity_metric = self.rigid_reg_kwargs[SIM_METRIC_KEY]
        transformer = self.rigid_reg_kwargs[TRANSFORMER_KEY]

        # print("\n======== Detecting features\n")
        rigid_registrar.generate_img_obj_list(feature_detector)

        if self.create_masks:
            # Remove feature points outside of mask
            for img_obj in rigid_registrar.img_obj_dict.values():
                slide_obj = self.get_slide(img_obj.name)
                features_in_mask_idx = warp_tools.get_xy_inside_mask(xy=img_obj.kp_pos_xy, mask=slide_obj.rigid_reg_mask)
                if len(features_in_mask_idx) > 0:
                    img_obj.kp_pos_xy = img_obj.kp_pos_xy[features_in_mask_idx, :]
                    img_obj.desc = img_obj.desc[features_in_mask_idx, :]


        # print("\n======== Matching images\n")
        if rigid_registrar.aleady_sorted:
            rigid_registrar.match_sorted_imgs(matcher, keep_unfiltered=False)

            for i, img_obj in enumerate(rigid_registrar.img_obj_list):
                img_obj.stack_idx = i

        else:
            rigid_registrar.match_imgs(matcher, keep_unfiltered=False)

            # print("\n======== Sorting images\n")
            rigid_registrar.build_metric_matrix(metric=similarity_metric)
            rigid_registrar.sort()

        rigid_registrar.distance_metric_name = matcher.metric_name
        rigid_registrar.distance_metric_type = matcher.metric_type
        rigid_registrar.get_iter_order()
        if rigid_registrar.size > 2:
            rigid_registrar.update_match_dicts_with_neighbor_filter(transformer, matcher)

        if self.reference_img_f is not None:
            ref_name = self.name_dict[self.reference_img_f]
        else:
            ref_name = valtils.get_name(rigid_registrar.reference_img_f)
            if self.do_rigid is not False:
                msg = " ".join([f"Best to specify `{REF_IMG_KEY}` when manually providing `{TFORM_MAT_KEY}`.",
                       f"Setting this image to be {ref_name}"])

                valtils.print_warning(msg)

        # Get output shapes #
        if tform_dict is None:
            named_tform_dict = {o.name: {"M":np.eye(3)} for o in rigid_registrar.img_obj_list}
        else:
            named_tform_dict = {valtils.get_name(k):v for k, v in tform_dict.items()}

        # Get output shapes #
        rigid_ref_obj = rigid_registrar.img_obj_dict[ref_name]
        ref_slide_obj = self.get_ref_slide()
        if ref_name in named_tform_dict.keys():
            ref_tforms = named_tform_dict[ref_name]
            if TFORM_SRC_SHAPE_KEY in ref_tforms:
                ref_tform_src_shape_rc = ref_tforms[TFORM_SRC_SHAPE_KEY]
            else:
                ref_tform_src_shape_rc = ref_slide_obj.slide_dimensions_wh[0][::-1]

            if TFORM_DST_SHAPE_KEY in ref_tforms:
                temp_out_shape_rc = ref_tforms[TFORM_DST_SHAPE_KEY]
            else:
                # Assume M was found by aligning to level 0 reference
                temp_out_shape_rc = ref_slide_obj.slide_dimensions_wh[0][::-1]

            ref_to_reg_sxy = (np.array(rigid_ref_obj.image.shape)/np.array(ref_tform_src_shape_rc))[::-1]
            out_rc = np.round(temp_out_shape_rc*ref_to_reg_sxy).astype(int)

        else:
            out_rc = rigid_ref_obj.image.shape

        scaled_M_dict = {}
        for img_name, img_tforms in named_tform_dict.items():
            matching_rigid_obj = rigid_registrar.img_obj_dict[img_name]
            matching_slide_obj = self.slide_dict[img_name]

            if TFORM_SRC_SHAPE_KEY in img_tforms:
                og_src_shape_rc = img_tforms[TFORM_SRC_SHAPE_KEY]
            else:
                og_src_shape_rc = matching_slide_obj.slide_dimensions_wh[0][::-1]

            temp_M = img_tforms[TFORM_MAT_KEY]
            if temp_M.shape[0] == 2:
                temp_M = np.vstack([temp_M, [0, 0, 1]])

            if TFORM_DST_SHAPE_KEY in img_tforms:
                og_dst_shape_rc = img_tforms[TFORM_DST_SHAPE_KEY]
            else:
                og_dst_shape_rc = ref_slide_obj.slide_dimensions_wh[0][::-1]

            img_corners_xy = warp_tools.get_corners_of_image(matching_rigid_obj.image.shape)[::-1]
            warped_corners = warp_tools.warp_xy(img_corners_xy, M=temp_M,
                                    transformation_src_shape_rc=og_src_shape_rc,
                                    transformation_dst_shape_rc=og_dst_shape_rc,
                                    src_shape_rc=matching_rigid_obj.image.shape,
                                    dst_shape_rc=out_rc)
            M_tform = transform.ProjectiveTransform()
            M_tform.estimate(warped_corners, img_corners_xy)
            for_reg_M = M_tform.params
            scaled_M_dict[matching_rigid_obj.name] = for_reg_M
            matching_rigid_obj.M = for_reg_M

        # Find M if not provided
        for moving_idx, fixed_idx in tqdm.tqdm(rigid_registrar.iter_order, desc=TRANSFORM_MSG, unit="image"):
            img_obj = rigid_registrar.img_obj_list[moving_idx]
            if img_obj.name in scaled_M_dict:
                continue

            prev_img_obj = rigid_registrar.img_obj_list[fixed_idx]
            img_obj.fixed_obj = prev_img_obj

            print(f"finding M for {img_obj.name}, which is being aligned to {prev_img_obj.name}")

            if fixed_idx == rigid_registrar.reference_img_idx:
                prev_M = np.eye(3)

            to_prev_match_info = img_obj.match_dict[prev_img_obj]
            src_xy = to_prev_match_info.matched_kp1_xy
            dst_xy = warp_tools.warp_xy(to_prev_match_info.matched_kp2_xy, prev_M)

            transformer.estimate(dst_xy, src_xy)
            img_obj.M = transformer.params

            prev_M = img_obj.M

        # Add registered image
        for img_obj in rigid_registrar.img_obj_list:
            img_obj.M_inv = np.linalg.inv(img_obj.M)

            img_obj.registered_img = warp_tools.warp_img(img=img_obj.image,
                                                        M=img_obj.M,
                                                        out_shape_rc=out_rc)

            img_obj.registered_shape_rc = img_obj.registered_img.shape[0:2]

        return rigid_registrar



    #En un principi aquí va el codi referent a la funció regid_register def rigid_register(self), però l'he hagut de ficar més a baix per tal de poder
    #utilitzar les funcions utilitzades en el non_rigid_registration i així poder fer ús de la bbox del non_rigid per calcular el IoU i la correlació 







    def micro_rigid_register(self):

        micro_rigid_registar = self.micro_rigid_registrar_cls(val_obj=self, **self.micro_rigid_registrar_params)
        micro_rigid_registar.register()

        # Draw in same order as regular rigid registration
        draw_list = [self.slide_dict[img_obj.name] for img_obj in self.rigid_registrar.img_obj_list]
        rigid_img_list = [slide_obj.warp_img(slide_obj.processed_img, non_rigid=False) for slide_obj in draw_list]
        self.micro_rigid_overlap_img = self.draw_overlap_img(rigid_img_list)

        micro_rigid_overlap_img_fout = os.path.join(self.overlap_dir, self.name + "_micro_rigid_overlap.png")
        warp_tools.save_img(micro_rigid_overlap_img_fout, self.micro_rigid_overlap_img, thumbnail_size=self.thumbnail_size)

        # Overwrite rigid registration results and update rigid registrar
        for slide_name, slide_obj in self.slide_dict.items():
            if not slide_obj.is_rgb:
                img_to_warp = slide_obj.processed_img
            else:
                img_to_warp = slide_obj.image
            img_to_warp = warp_tools.resize_img(img_to_warp, slide_obj.processed_img_shape_rc)
            warped_img = slide_obj.warp_img(img_to_warp, non_rigid=False, crop=self.crop)
            warp_tools.save_img(slide_obj.rigid_reg_img_f, warped_img.astype(np.uint8), thumbnail_size=self.thumbnail_size)

            if slide_obj.fixed_slide is None:
                continue
            fixed_slide = slide_obj.fixed_slide
            fixed_rigid_obj = self.rigid_registrar.img_obj_dict[fixed_slide.name]

            rigid_img_obj = self.rigid_registrar.img_obj_dict[slide_obj.name]
            rigid_img_obj.M = slide_obj.M
            rigid_img_obj.M_inv = np.linalg.inv(slide_obj.M)
            rigid_img_obj.registered_img = slide_obj.warp_img(img_to_warp, non_rigid=False, crop=False)

            rigid_img_obj.match_dict[fixed_rigid_obj].matched_kp1_xy = slide_obj.xy_matched_to_prev
            rigid_img_obj.match_dict[fixed_rigid_obj].matched_kp2_xy = slide_obj.xy_in_prev
            rigid_img_obj.match_dict[fixed_rigid_obj].n_matches = slide_obj.xy_in_prev.shape[0]

            fixed_rigid_obj.match_dict[rigid_img_obj].matched_kp1_xy = slide_obj.xy_in_prev
            fixed_rigid_obj.match_dict[rigid_img_obj].matched_kp2_xy = slide_obj.xy_matched_to_prev
            fixed_rigid_obj.match_dict[rigid_img_obj].n_matches = slide_obj.xy_in_prev.shape[0]

    def draw_matches(self, dst_dir):
        """Draw and save images of matching features

        Parameters
        ----------
        dst_dir : str
            Where to save the images of the matched features
        """

        dst_dir = str(dst_dir)
        pathlib.Path(dst_dir).mkdir(exist_ok=True, parents=True)

        slide_idx, slide_names = list(zip(*[[slide_obj.stack_idx, slide_obj.name] for slide_obj in self.slide_dict.values()]))
        slide_order = np.argsort(slide_idx) # sorts ascending
        slide_list = [self.slide_dict[slide_names[i]] for i in slide_order]
        for moving_idx, fixed_idx in self.iter_order:
            moving_slide = slide_list[moving_idx]
            fixed_slide = slide_list[fixed_idx]

            # RGB draw images
            if moving_slide.image.ndim == 3 and moving_slide.is_rgb:
                moving_draw_img = warp_tools.resize_img(moving_slide.image, moving_slide.processed_img.shape[0:2])
            else:
                moving_draw_img = moving_slide.processed_img

            if fixed_slide.image.ndim == 3 and fixed_slide.is_rgb:
                fixed_draw_img = warp_tools.resize_img(fixed_slide.image, fixed_slide.processed_img.shape[0:2])
            else:
                fixed_draw_img = fixed_slide.processed_img

            all_matches_img = viz.draw_matches(src_img=moving_draw_img, kp1_xy=moving_slide.xy_matched_to_prev,
                                               dst_img=fixed_draw_img,  kp2_xy=moving_slide.xy_in_prev,
                                               rad=3, alignment='horizontal')
            matches_f_out = os.path.join(dst_dir, f"{self.name}_{moving_slide.name}_to_{fixed_slide.name}_matches.png")
            warp_tools.save_img(matches_f_out, all_matches_img)

    def create_non_rigid_reg_mask(self):
        """
        Get mask for non-rigid registration
        """

        if self.create_masks:
            #Codi meu
            #print(f"Entra aquí")
            #
            non_rigid_mask = self._create_mask_from_processed()

            #Codi meu
            #non_rigid_mask_path = os.path.join(self.mask_dir, "_non_rigid_mask_bbox.png")
            #warp_tools.save_img(non_rigid_mask_path, non_rigid_mask)
            #
            #Retorna una màscara de 1673x1649 (mida gran de les màscares del rígid) on hi ha la bbox del rigid amb blanc i el fons amb negre

        else:
            non_rigid_mask = self._create_non_rigid_reg_mask_from_bbox()

        for slide_obj in self.slide_dict.values():
            slide_obj.non_rigid_reg_mask = non_rigid_mask

        # Save thumbnail of mask
        ref_slide = self.get_ref_slide()
        if ref_slide.img_type == slide_tools.IHC_NAME:
            #Codi meu
            #print(f"Entra aquí")
            #
            ref_img = warp_tools.resize_img(ref_slide.image, ref_slide.processed_img_shape_rc)
            warped_ref_img = ref_slide.warp_img(ref_img, non_rigid=False, crop=CROP_REF)
            #Codi meu
            #ref_img_path = os.path.join(self.mask_dir, "_ref_img_non_rigid.png")
            #warp_tools.save_img(ref_img_path, ref_img)
            #non_rigid_mask_path = os.path.join(self.mask_dir, "_warped_ref_img.png")
            #warp_tools.save_img(non_rigid_mask_path, warped_ref_img)
            #
            #Són les dos imatges iguals i és la imatge de AR22B050886B0010_non_rigid_mask.png però sense la bbox dibuixada

        else:
            warped_ref_img = ref_slide.warp_img(ref_slide.processed_img, non_rigid=False, crop=CROP_REF)

        pathlib.Path(self.mask_dir).mkdir(exist_ok=True, parents=True)
        thumbnail_img = self.create_thumbnail(warped_ref_img)

        draw_mask = warp_tools.resize_img(non_rigid_mask, ref_slide.reg_img_shape_rc, interp_method="nearest")
        _, overlap_mask_bbox_xywh = self.get_crop_mask(CROP_REF)
        #Codi meu
        #print(f"Variable overlap_mask_bbox_xywh: {overlap_mask_bbox_xywh}")
        #

        draw_mask = warp_tools.crop_img(draw_mask, overlap_mask_bbox_xywh.astype(int))
        thumbnail_mask = self.create_thumbnail(draw_mask)

        thumbnail_mask_outline = viz.draw_outline(thumbnail_img, thumbnail_mask)
        outline_f_out = os.path.join(self.mask_dir_non_rigid_registration, f'{self.name}_non_rigid_mask.png')
        warp_tools.save_img(outline_f_out, thumbnail_mask_outline)

    def _create_non_rigid_reg_mask_from_bbox(self, slide_list=None):
        """Mask will be bounding box of image overlaps

        """
        ref_slide = self.get_ref_slide()
        combo_mask = np.zeros(ref_slide.reg_img_shape_rc, dtype=int)

        if slide_list is None:
            slide_list = list(self.slide_dict.values())

        for slide_obj in slide_list:
            img_bbox = np.full(slide_obj.processed_img_shape_rc, 255, dtype=np.uint8)
            rigid_mask = slide_obj.warp_img(img_bbox, non_rigid=False, crop=False, interp_method="nearest")
            combo_mask[rigid_mask > 0] += 1

        n = len(slide_list)
        overlap_mask = (combo_mask == n).astype(np.uint8)
        overlap_bbox = warp_tools.xy2bbox(warp_tools.mask2xy(overlap_mask))
        c0, r0 = overlap_bbox[:2]
        c1, r1 = overlap_bbox[:2] + overlap_bbox[2:]

        non_rigid_mask = np.zeros_like(overlap_mask)
        non_rigid_mask[r0:r1, c0:c1] = 255

        return non_rigid_mask

    def _create_mask_from_processed(self, slide_list=None):

        combo_mask = np.zeros(self.aligned_img_shape_rc, dtype=int)

        if slide_list is None:
            slide_list = list(self.slide_dict.values())

        for i, slide_obj in enumerate(self.slide_dict.values()):
            rigid_mask = slide_obj.warp_img(slide_obj.rigid_reg_mask, non_rigid=False, crop=False, interp_method="nearest")
            combo_mask[rigid_mask > 0] += 1

        temp_non_rigid_mask = 255*filters.apply_hysteresis_threshold(combo_mask, 0.5, self.size-0.5).astype(np.uint8)
        overlap_mask = 255*ndimage.binary_fill_holes(temp_non_rigid_mask).astype(np.uint8)

        to_combine_list = [None] * len(slide_list)
        for i, slide_obj in enumerate(slide_list):
            for_summary = exposure.rescale_intensity(slide_obj.warp_img(slide_obj.processed_img, non_rigid=False, crop=False), out_range=(0,1))
            to_combine_list[i] = for_summary

        combo_img = np.dstack(to_combine_list)
        summary_img = np.median(combo_img, axis=2)
        summary_img[overlap_mask == 0] = 0

        low_t, high_t = filters.threshold_multiotsu(summary_img[overlap_mask > 0])
        fg = 255*filters.apply_hysteresis_threshold(summary_img, low_t, high_t).astype(np.uint8)
        fg_bbox_mask = np.zeros_like(overlap_mask)
        fg_bbox = warp_tools.xy2bbox(warp_tools.mask2xy(fg))
        c0, r0 = fg_bbox[0:2]
        c1, r1 = fg_bbox[0:2] + fg_bbox[2:]

        #Codi meu que augment la bbox en 20 pixels (Cal acordar quin és el padding correcte)
        padding = 20
        r0 = max(r0 - padding, 0)
        c0 = max(c0 - padding, 0)
        r1 = min(r1 + padding, overlap_mask.shape[0])
        c1 = min(c1 + padding, overlap_mask.shape[1])
        fg_bbox_mask[r0:r1, c0:c1] = 255

        return fg_bbox_mask

    def _create_non_rigid_reg_mask_from_rigid_masks(self, slide_list=None):
        """
        Get mask that will cover all tissue. Use hysteresis thresholding to ignore
        masked regions found in only 1 image.

        """

        if slide_list is None:
            slide_list = list(self.slide_dict.values())

        combo_mask = np.zeros(self.aligned_img_shape_rc, dtype=int)
        for i, slide_obj in enumerate(slide_list):
            rigid_mask = slide_obj.warp_img(slide_obj.rigid_reg_mask, non_rigid=False, crop=False, interp_method="nearest")
            combo_mask[rigid_mask > 0] += 1

        temp_mask = 255*filters.apply_hysteresis_threshold(combo_mask, 0.5, self.size-0.5).astype(np.uint8)

        # Draw convex hull around each region
        final_mask = 255*ndimage.binary_fill_holes(temp_mask).astype(np.uint8)
        final_mask = preprocessing.mask2contours(final_mask)

        return final_mask

    def pad_displacement(self, dxdy, out_shape_rc, bbox_xywh):

        is_array = not isinstance(dxdy, pyvips.Image)
        if is_array:
            vips_dxdy = warp_tools.numpy2vips(np.dstack(dxdy))
        else:
            vips_dxdy = dxdy

        full_dxdy = pyvips.Image.black(out_shape_rc[1], out_shape_rc[0], bands=2).cast("float")
        full_dxdy = full_dxdy.insert(vips_dxdy, *bbox_xywh[0:2])

        if is_array:
            full_dxdy = warp_tools.vips2numpy(full_dxdy)
            full_dxdy = np.array([full_dxdy[..., 0], full_dxdy[..., 1]])

        return full_dxdy

    def get_nr_tiling_params(self, non_rigid_registrar_cls,
                             processor_dict,
                             img_specific_args,
                             tile_wh):
        """Get extra parameters need for tiled non-rigid registration

        processor_dict : dict
            Each key should be the filename of the image, and the value either a subclassed
            preprocessing.ImageProcessor, or a list, where the 1st element is the processor,
            and the second element a dictionary of keyword arguments passed to the processor.
            If `None`, then a default processor will be used for each image based on
            the inferred modality.
        """
        if img_specific_args is None:
            img_specific_args = {}

        for slide_obj in self.slide_dict.values():

            processing_cls, processing_kwargs = processor_dict[slide_obj.name]
            # Add registration parameters
            tiled_non_rigid_reg_params = {}
            tiled_non_rigid_reg_params[non_rigid_registrars.NR_CLS_KEY] = non_rigid_registrar_cls
            if self.norm_method is not None:
                tiled_non_rigid_reg_params[non_rigid_registrars.NR_STATS_KEY] = self.target_processing_stats
            tiled_non_rigid_reg_params[non_rigid_registrars.NR_TILE_WH_KEY] = tile_wh

            tiled_non_rigid_reg_params[non_rigid_registrars.NR_PROCESSING_CLASS_KEY] = processing_cls
            tiled_non_rigid_reg_params[non_rigid_registrars.NR_PROCESSING_KW_KEY] = processing_kwargs
            tiled_non_rigid_reg_params[non_rigid_registrars.NR_PROCESSING_INIT_KW_KEY] = {"src_f": slide_obj.src_f,
                                                                                          "series": slide_obj.series,
                                                                                          "reader": deepcopy(slide_obj.reader)
                                                                                          }
            img_specific_args[slide_obj.name] = tiled_non_rigid_reg_params

        non_rigid_registrar_cls = non_rigid_registrars.NonRigidTileRegistrar

        return non_rigid_registrar_cls, img_specific_args

    def prep_images_for_large_non_rigid_registration(self, max_img_dim,
                                                     processor_dict,
                                                     updating_non_rigid=False,
                                                     mask=None):

        """Scale and process images for non-rigid registration using larger images

        Parameters
        ----------
        max_img_dim : int, optional
            Maximum size of image to be used for non-rigid registration. If None, the whole image
            will be used  for non-rigid registration

        processor_dict : dict
            Each key should be the filename of the image, and the value either a subclassed
            preprocessing.ImageProcessor, or a list, where the 1st element is the processor,
            and the second element a dictionary of keyword arguments passed to the processor.
            If `None`, then a default processor will be used for each image based on
            the inferred modality.

        updating_non_rigid : bool, optional
            If `True`, the slide's current non-rigid registration will be applied
            The new displacements found using these larger images can therefore be used
            to update existing dxdy. If `False`, only the rigid transform will be applied,
            so this will be the first non-rigid transformation.

        mask : ndarray, optional
            Binary image indicating where to perform the non-rigid registration. Should be
            based off an already registered image.

        Returns
        -------
        img_dict : dictionary
            Dictionary that can be passed to a non-rigid registrar

        max_img_dim : int
            Maximum size of image to do non-rigid registration on. May be different
            if the requested size was too big

        scaled_non_rigid_mask : ndarray
            Scaled mask to use for non-rigid registration

        full_out_shape : ndarray of int
            Shape (row, col) of the warped images, without cropping

        mask_bbox_xywh : list
            Bounding box of `mask`. If `mask` is None, then so will `mask_bbox_xywh`

        """

        warp_full_img = max_img_dim is None
        if not warp_full_img:
            all_max_dims = [np.any(np.max(slide_obj.slide_dimensions_wh, axis=1) >= max_img_dim) for slide_obj in self.slide_dict.values()]
            #Codi meu
            #print(f'Dimensions primer print: {all_max_dims}')
            #Entra, [True, True]

            if not np.all(all_max_dims):
                #Codi meu
                #print('2')
                #No entra
                img_maxes = [np.max(slide_obj.slide_dimensions_wh, axis=1)[0] for slide_obj in self.slide_dict.values()]
                smallest_img_max = np.min(img_maxes)
                msg = (f"Requested size of images for non-rigid registration was {max_img_dim}. "
                    f"However, not all images are this large. Setting `max_non_rigid_registration_dim_px` to "
                    f"{smallest_img_max}, which is the largest dimension of the smallest image")
                valtils.print_warning(msg)
                max_img_dim = smallest_img_max

        ref_slide = self.get_ref_slide()

        max_s = np.min(ref_slide.slide_dimensions_wh[0]/np.array(ref_slide.processed_img_shape_rc[::-1]))
        if mask is None:
            if warp_full_img:
                #Codi meu
                #print('3')
                #No entra
                s = max_s
            else:
                #Codi meu
                #print('4')
                #No entra
                s = np.min(max_img_dim/np.array(ref_slide.processed_img_shape_rc))
        else:
            # Determine how big image would have to be to get mask with maxmimum dimension = max_img_dim
            if isinstance(mask, pyvips.Image):
                #Codi meu
                #print('5')
                #No entra
                mask_shape_rc = np.array((mask.height, mask.width))
            else:
                mask_shape_rc = np.array(mask.shape[0:2]) #Agafa les dimensions de la màscara
                #Codi meu
                #print(f'Mida de la mascara mask, print 6: {mask_shape_rc}')
                #path_slide_mask_8 = os.path.join(self.mask_dir_non_rigid_registration, "Slide_mask_8.png")
                #warp_tools.save_img(path_slide_mask_8, mask)
                #La màscara mask és una màscara de mida [1673, 1649] que té la bbox de color blanc i el fons negre. La bbox està centrada al mig i és lo mateix 
                #que la imatge dels passos intermitjos del registre rigid.
                #Entra, [1673, 1649]

            to_reg_mask_sxy = (mask_shape_rc/np.array(ref_slide.reg_img_shape_rc))[::-1]
            if not np.all(to_reg_mask_sxy == 1): #Comprova si la màscara i la imatge estan a la mateixa escala, com que ho estan no cal canviar res
                #Codi meu
                #print('7')
                #No entra
                # Resize just in case it's huge. Only need bounding box
                reg_size_mask = warp_tools.resize_img(mask, ref_slide.reg_img_shape_rc, interp_method="nearest")
            else:
                reg_size_mask = mask
                #Codi meu
                #print('8')
                #Entra
            reg_size_mask_xy = warp_tools.mask2xy(reg_size_mask) #Retorna la bbox
            to_reg_mask_bbox_xywh = list(warp_tools.xy2bbox(reg_size_mask_xy)) #Retorna la bbox
            to_reg_mask_wh = np.round(to_reg_mask_bbox_xywh[2:]).astype(int)
            #print(f'Dimensions de la bbox: {to_reg_mask_wh}')
            #Dim de la bbox és de [304, 302]
            #S'han extret les coordenades (x,y) de pixels positius de la màscara, calcula el bbox i les dimensions
            if warp_full_img:
                #Codi meu
                #print('9')
                #No entra
                s = max_s
            else:
                #Codi meu
                #print('10')
                #Entra
                s = np.min(max_img_dim/np.array(to_reg_mask_wh))
                #print(f'Factor de escalat: {s}')
                # s = 6,0855
                #Calcula el factor d'escala per fer cabre la bbox dins de max_img_dim

        if s < max_s:
            full_out_shape = self.get_aligned_slide_shape(s)
            #Codi meu
            #print(f'Shape després del factor de escalat {full_out_shape}')
            #Entra [10036, 10182] = factos d'escalat s (6,0855) * mask_shape_rc ([1649, 1673])
            #Calcula la shape de la imatge que resultarà després d'escalar amb el factor s de dalt
        else:
            #Codi meu
            #print('12')
            #No entra
            full_out_shape = self.get_aligned_slide_shape(0)

        if mask is None:
            #Codi meu
            #print('13')
            #No entra
            out_shape = full_out_shape
            mask_bbox_xywh = None
        else:
            # If masking, the area will be smaller. Get bounding box
            mask_sxy = (full_out_shape/mask_shape_rc)[::-1]
            mask_bbox_xywh = list(warp_tools.xy2bbox(mask_sxy*reg_size_mask_xy))
            mask_bbox_xywh[2:] = np.round(mask_bbox_xywh[2:]).astype(int)
            out_shape = mask_bbox_xywh[2:][::-1]
            #Codi meu
            #print(f'Out shape de la bbox després de la escala: {out_shape}')
            #Entra [1850, 1838] = factor d'escalat s (6,0855) * Dim de la bbox ([304, 302])
            #Calcula de bbox adaptat a l'escala final i shape del crop

            if not isinstance(mask, pyvips.Image):
                #Codi meu
                #print('15')
                #Entra
                vips_micro_reg_mask = warp_tools.numpy2vips(mask)
            else:
                #Codi meu
                #print('16')
                #No entra
                vips_micro_reg_mask = mask
            vips_micro_reg_mask = warp_tools.resize_img(vips_micro_reg_mask, full_out_shape, interp_method="nearest")
            vips_micro_reg_mask = warp_tools.crop_img(img=vips_micro_reg_mask, xywh=mask_bbox_xywh)
            #Els 3 vips... lo que fan és convertir la màscara numpy a pyvips, li fa un resize i fa un crop segons bbox

        if ref_slide.reader.metadata.bf_datatype is not None:
            #Codi meu
            #print('17')
            #Entra
            np_dtype = slide_tools.BF_FORMAT_NUMPY_DTYPE[ref_slide.reader.metadata.bf_datatype]
        else:
            #Codi meu
            #print('18')
            #No entra
            # Assuming images not read by bio-formats are RGB read using from openslide or png, jpeg, etc...
            np_dtype = "uint8"

        displacement_gb = self.size*warp_tools.calc_memory_size_gb(full_out_shape, 2, "float32")
        processed_img_gb = self.size*warp_tools.calc_memory_size_gb(out_shape, 1, "uint8")
        img_gb = self.size*warp_tools.calc_memory_size_gb(out_shape, ref_slide.reader.metadata.n_channels, np_dtype)
        #Les 3 linies de dalt calculen quanta memoria ocuparan els diferents elements involucrats del reg_no_rigid

        # Size of full displacement fields, all larger processed images, and an image that will be processed
        estimated_gb = img_gb + displacement_gb + processed_img_gb
        use_tiler = False
        if estimated_gb > TILER_THRESH_GB:
            #Codi meu
            #print('19')
            #No entra
            # Avoid having huge displacement fields saved in registrar.
            use_tiler = True

        scaled_warped_img_list = [None] * self.size
        scaled_mask_list = [None] * self.size
        img_names_list = [None] * self.size
        img_f_list = [None] * self.size

        # print("\n======== Preparing images for non-rigid registration\n")
        #Codi meu
        iter_count = 0
        #
        for slide_obj in tqdm.tqdm(self.slide_dict.values(), desc=PREP_NON_RIGID_MSG, unit="image"):
            # Get image to warp. Likely a larger image scaled down to specified shape #
            src_img_shape_rc, src_M = warp_tools.get_src_img_shape_and_M(transformation_src_shape_rc=slide_obj.processed_img_shape_rc,
                                                                            transformation_dst_shape_rc=slide_obj.reg_img_shape_rc,
                                                                            dst_shape_rc=full_out_shape,
                                                                            M=slide_obj.M)

            if max_img_dim is not None:
                closest_img_levels = np.where(np.max(slide_obj.slide_dimensions_wh, axis=1) < np.max(src_img_shape_rc))[0]
                if len(closest_img_levels) > 0:
                    #Codi meu
                    #print('20')
                    #Entra
                    closest_img_level = closest_img_levels[0] - 1
                    #Aquí desideix quin nivell de la piràmide d'imatge ha de fer servir (nivell més gran que càpiga)
                    print(f'Nivell de la piramide: {closest_img_level}')
                    #Nivell de la piràmide és el nivell 5, imatge de [9192, 8504]
                else:
                    #Codi meu
                    print('21')
                    #No entra
                    closest_img_level = len(slide_obj.slide_dimensions_wh) - 1
                    print(f'Nivell de la piramide: {closest_img_level}')
            else:
                #Codi meu
                print('22')
                #No entra
                closest_img_level = 0

            vips_level_img = slide_obj.slide2vips(closest_img_level)
            

            #Codi meu
            if iter_count == 0:
                imatge_HE_etiquetes_sense_padding = self.eliminar_padding_calculat(path_HE_original = slide_obj.src_f, level = closest_img_level)
                self.preparar_imatge_HE_etiquetes(path_HE_original = slide_obj.src_f, imatge_HE_etiquetes_sense_padding = imatge_HE_etiquetes_sense_padding, level=closest_img_level, rigid_or_non_rigid = 1)

            path_he_etiquetes = os.path.join(self.mask_dir_non_rigid_registration, "HE_etiquetes_Valis_non_register.png")
            he_etiquetes = imread(path_he_etiquetes)
            
            #MOLT IMPORTANT: ESTEM DINS UN BUCLE FOR, si només es guarden les imatges, només es guardaran les de IHC pq es la segona execució i 
            #sobreescriuen les imatges de HE guardades en la primera execució
            
            #print(f'Mirem les vegades que passa per aquí: {vips_level_img}')
            #Codi meu
            path_slide_mask_9 = os.path.join(self.mask_dir_non_rigid_registration, str(iter_count) + "_Slide_mask_9.png")
            warp_tools.save_img(path_slide_mask_9, vips_level_img)
            #


            img_to_warp = warp_tools.resize_img(vips_level_img, src_img_shape_rc)
            #Codi meu
            if iter_count == 0: he_etiquetes_1 = warp_tools.resize_img(he_etiquetes, src_img_shape_rc)

            #Carrega la imatge del nivell triat i li fa un resize
            #Codi meu
            print(f'Shape de la vips_level_img: {src_img_shape_rc}')
            #El src_img_shape_rc és de [6469. 6993]

            if updating_non_rigid:
                #Codi meu
                #print('23')
                #No entra
                dxdy = slide_obj.bk_dxdy
            else:
                #Codi meu
                #print('24')
                #Entra
                dxdy = None

            # Get mask covering tissue
            temp_slide_mask = slide_obj.warp_img(slide_obj.rigid_reg_mask, non_rigid=dxdy is not None, crop=False, interp_method="nearest")
            temp_slide_mask = warp_tools.numpy2vips(temp_slide_mask)
            slide_mask = warp_tools.resize_img(temp_slide_mask, full_out_shape, interp_method="nearest")
            #Codi meu
            #path_slide_mask_1 = os.path.join(self.mask_dir, "Slide_mask_1.png")
            #warp_tools.save_img(path_slide_mask_1, slide_mask)
            #És la mascàra on hi ha el teixit d'IHC al mig de la imatge (en blanc i el fons negre), és com si fos la màscara que hi havia als 
            #passos intermitjos del serial_rigid abans d'aplicar la bbox al centre de la imatge per quedar-nos amb el teixit a registrar,
            #però amb resolució corresponent al full_out_shape que és de [10182, 10036]

            if mask_bbox_xywh is not None:
                #Codi meu
                #print('25')
                #Entra
                slide_mask = warp_tools.crop_img(slide_mask, mask_bbox_xywh)
                #Aplica el registre rígid a la màscara, li fa un resize i un crop segons bbox

            #Codi meu
            #path_slide_mask_2 = os.path.join(self.mask_dir, "Slide_mask_2.png")
            #warp_tools.save_img(path_slide_mask_2, slide_mask)
            #És la màscara que hem guardat a dalt amb Slide_mask_1 però amb la bbox aplicada a la zona del teixit i havent-li fet un zoom a la part de la bbox,
            #fent que la imatge quedi de la resolució out_shape de [1850, 1838]

            # Get mask that covers image
            temp_processing_mask = pyvips.Image.black(img_to_warp.width, img_to_warp.height).invert()
            #Codi meu
            #path_slide_mask_3_1 = os.path.join(self.mask_dir_non_rigid_registration, "Slide_mask_3_1.png")
            #warp_tools.save_img(path_slide_mask_3_1, temp_processing_mask)
            #
            processing_mask = warp_tools.warp_img(img=temp_processing_mask, M=slide_obj.M,
                bk_dxdy=dxdy,
                transformation_src_shape_rc=slide_obj.processed_img_shape_rc,
                transformation_dst_shape_rc=slide_obj.reg_img_shape_rc,
                out_shape_rc=full_out_shape,
                bbox_xywh=mask_bbox_xywh,
                interp_method="nearest")
            #Codi meu
            #path_slide_mask_3 = os.path.join(self.mask_dir_non_rigid_registration, "Slide_mask_3.png")
            #warp_tools.save_img(path_slide_mask_3, processing_mask)
            #Crea una màscara per saber quina part real ocupa la imatge a l'espai registrat. Bàsicament és una imatge de la mida d'out_shape (mida de la bbox)
            #on hauria de modificar la mida si no hi hagués teixits, no ho fa i la mida és la mateixa, per tant és una imatge de tot blanc i de mida out_shape  

            if not use_tiler:
                #Codi meu
                #print('26')
                #Entra
                # Process image using same method for rigid registration #
                unprocessed_warped_img = warp_tools.warp_img(img=img_to_warp, M=slide_obj.M,
                    bk_dxdy=dxdy,
                    transformation_src_shape_rc=slide_obj.processed_img_shape_rc,
                    transformation_dst_shape_rc=slide_obj.reg_img_shape_rc,
                    out_shape_rc=full_out_shape,
                    bbox_xywh=mask_bbox_xywh,
                    bg_color=slide_obj.bg_color)

                #Codi meu
                if iter_count == 0: 
                    he_etiquetes_2 = warp_tools.warp_img(img=he_etiquetes_1, M=slide_obj.M,
                                                                bk_dxdy=dxdy,
                                                                transformation_src_shape_rc=slide_obj.processed_img_shape_rc,
                                                                transformation_dst_shape_rc=slide_obj.reg_img_shape_rc,
                                                                out_shape_rc=full_out_shape,
                                                                bbox_xywh=mask_bbox_xywh,
                                                                bg_color=slide_obj.bg_color)
                    path_img_he_etiquetes_registering = os.path.join(self.mask_dir_non_rigid_registration, "HE_etiquetes_Valis_non_registering.png")
                    warp_tools.save_img(path_img_he_etiquetes_registering, he_etiquetes_2)

                    #out_shape_rc_original_he_etiquetes = he_etiquetes.shape  # (height, width)
                    #print(f'Mida he_etiquetes: {out_shape_rc_original_he_etiquetes}')
                    #he_etiquetes_reescalada = warp_tools.resize_img(he_etiquetes_2, out_shape_rc_original_he_etiquetes)
                    #path_img_he_etiquetes_registering_1 = os.path.join(self.mask_dir_non_rigid_registration, "HE_etiquetes_Valis_non_registering_mida_gran.png")
                    #warp_tools.save_img(path_img_he_etiquetes_registering_1, he_etiquetes_reescalada)
                #

                #Codi meu
                path_slide_mask_4 = os.path.join(self.mask_dir_non_rigid_registration, str(iter_count) + "_Slide_mask_4.png")
                warp_tools.save_img(path_slide_mask_4, unprocessed_warped_img)
                #Es la imatge Slide_mask_2 però per contres de ser una màscara amb teixit blanc i fons negre com és la _2, aquesta té tot el teixit de la IHC
                #que es veu i el fons blanc
                unprocessed_warped_img = warp_tools.vips2numpy(unprocessed_warped_img)

                processing_cls, processing_kwargs = processor_dict[slide_obj.name]
                processor = processing_cls(image=unprocessed_warped_img,
                                           src_f=slide_obj.src_f,
                                           level=closest_img_level,
                                           series=slide_obj.series,
                                           reader=slide_obj.reader)

                try:
                    processed_img = processor.process_image(**processing_kwargs)
                    #Codi meu
                    #print(f'Entra aqui')
                    #Entra
                except TypeError:
                    # processor.process_image doesn't take kwargs
                    processed_img = processor.process_image()
                processed_img = exposure.rescale_intensity(processed_img, out_range=(0, 255)).astype(np.uint8)

                np_mask = warp_tools.vips2numpy(slide_mask)
                processed_img[np_mask == 0] = 0
                #S'aplica el registre rígid, es processa la imatge, normalitza i es deixen fora les zones buides
                # Normalize images using stats collected for rigid registration #
                if self.norm_method is not None:
                    #Codi meu
                    #print('27')
                    #Entra
                    processed_img = preprocessing.norm_img_stats(img=processed_img, target_stats=self.target_processing_stats, mask=np_mask)
                    #Codi meu
                    path_slide_mask_5 = os.path.join(self.mask_dir_non_rigid_registration, str(iter_count) + "_Slide_mask_5.png")
                    warp_tools.save_img(path_slide_mask_5, processed_img)
                    #Processed_img és el resultat d'aplicar una normalització estadística per tal de que l'histograma de la imatge d'IHC sigui més uniforme i es 
                    #pugui aplicar més bé les transformacions que es faran a posterior. És com si fes l'equalització de l'histograma però d'una forma específica per aquest teixit
                    #Sin màs tmp té massa importància lo que es veu a la imatge, només cal saber que és el resultat d'aplicar l'equalització
                warped_img = exposure.rescale_intensity(processed_img, out_range=(0, 255)).astype(np.uint8)

            else:
                if not warp_full_img:
                    #Codi meu
                    #print('28')
                    #No entra
                    warped_img = warp_tools.warp_img(img=img_to_warp, M=slide_obj.M,
                                bk_dxdy=dxdy,
                                transformation_src_shape_rc=slide_obj.processed_img_shape_rc,
                                transformation_dst_shape_rc=slide_obj.reg_img_shape_rc,
                                out_shape_rc=full_out_shape,
                                bbox_xywh=mask_bbox_xywh)
                else:
                    #Codi meu
                    #print('29')
                    #No entra
                    warped_img = slide_obj.warp_slide(0, non_rigid=updating_non_rigid, crop=mask_bbox_xywh)

            # Get mask #
            if mask is not None:
                #Codi meu
                #print('30')
                #Entra
                slide_mask = (vips_micro_reg_mask==0).ifthenelse(0, slide_mask) #On no hi ha màscara global es posa a 0
                #Codi meu
                path_slide_mask_6 = os.path.join(self.mask_dir_non_rigid_registration, str(iter_count) + "_Slide_mask_6.png")
                warp_tools.save_img(path_slide_mask_6, slide_mask)
                #És la mateixa màscara que la slide_mask_2
            # Update lists
            img_f_list[slide_obj.stack_idx] = slide_obj.src_f
            img_names_list[slide_obj.stack_idx] = slide_obj.name
            scaled_warped_img_list[slide_obj.stack_idx] = warped_img
            scaled_mask_list[slide_obj.stack_idx] = processing_mask


            #Codi meu
            iter_count += 1
            #


        img_dict = {serial_non_rigid.IMG_LIST_KEY: scaled_warped_img_list,
                    serial_non_rigid.IMG_F_LIST_KEY: img_f_list,
                    serial_non_rigid.MASK_LIST_KEY: scaled_mask_list,
                    serial_non_rigid.IMG_NAME_KEY: img_names_list
                    }

        if ref_slide.non_rigid_reg_mask is not None:
            #Codi meu
            #print('31')
            #Entra
            vips_nr_mask = warp_tools.numpy2vips(ref_slide.non_rigid_reg_mask)
            scaled_non_rigid_mask = warp_tools.resize_img(vips_nr_mask, full_out_shape, interp_method="nearest")
            #Es converteix la non-rigid mask existent a pyvips i li fa un resize
            if mask is not None:
                #Codi meu
                #print('32')
                #Entra
                scaled_non_rigid_mask = scaled_non_rigid_mask.extract_area(*mask_bbox_xywh)
                scaled_non_rigid_mask = (vips_micro_reg_mask == 0).ifthenelse(0, scaled_non_rigid_mask)
                #Codi meu
                path_slide_mask_7 = os.path.join(self.mask_dir_non_rigid_registration, "Slide_mask_7.png")
                warp_tools.save_img(path_slide_mask_7, scaled_non_rigid_mask)
                #Fa un crop de la màscara non-rigid existent i la filtra segons la global
                #És la mateixa màscara que la Slide_mask_3
            if not use_tiler:
                #Codi meu
                #print('33')
                #Entra
                scaled_non_rigid_mask = warp_tools.vips2numpy(scaled_non_rigid_mask)
                #Passa la màscara de pyvips a numpy
        else:
            #Codi meu
            #print('34')
            #No entra
            scaled_non_rigid_mask = None

        if mask is not None:
            #Codi meu
            #print('35')
            #Entra
            final_max_img_dim = np.max(mask_bbox_xywh[2:])
            #Codi meu
            #print(f'Mida final del preprocessing del non_rigid: {final_max_img_dim}')
            #Printeja 1850
            #S'actualitza la mida màxima real en base al bbox utilitzat
        else:
            #Codi meu
            #print('36')
            #No entra
            final_max_img_dim = max_img_dim

        return img_dict, final_max_img_dim, scaled_non_rigid_mask, full_out_shape, mask_bbox_xywh, use_tiler




    def rigid_register(self):
        """Rigidly register slides

        Also saves thumbnails of rigidly registered images.

        Returns
        -------
        rigid_registrar : SerialRigidRegistrar
            SerialRigidRegistrar object that performed the rigid registration.

        """

        if self.denoise_rigid:
            self.denoise_images()
            #Codi meu
            #print(f"Màscara de teixit geenrada per {slide_obj.name}, dimensions: {tissue_mask.shape}")
            #
        
        #Codi meu
        #print("processed dir:")
        #print(self.processed_dir)
        #


        print("\n==== Rigid registration\n")
        if self.do_rigid is True:
            rigid_registrar = serial_rigid.register_images(self.mask_dir_rigid_registration, self.mask_dir_rigid_serial_rigid, img_dir=self.processed_dir,
                                                           align_to_reference=self.align_to_reference,
                                                           valis_obj=self,
                                                           **self.rigid_reg_kwargs)
            #Codi meu: He afegit la variable mask_dir al codi serial_rigid.register_images i li passo self.mask_dir pq pugui accedir a la màscara Màscara_teixit_HE_original.png
            #i poder aplicar les diferents matrius que té el serial_rigid per poder-les aplicar jo després a mà quan vulgui fer el registre rígid de les màscares per calcular el IoU
        else:
            if isinstance(self.do_rigid, dict):
                # User provided transforms
                rigid_tforms = self.do_rigid
            elif self.do_rigid is False:
                # Skip rigid registration
                rigid_tforms = None

            rigid_registrar = self.rigid_register_partial(tform_dict=rigid_tforms)

        self.end_rigid_time = time()
        self.rigid_registrar = rigid_registrar

        if rigid_registrar is False:
            msg = "Rigid registration failed"
            valtils.print_warning(msg, rgb=Fore.RED)

            return False

        # Draw and save overlap image #
        self.aligned_img_shape_rc = rigid_registrar.img_obj_list[0].registered_shape_rc
        #print(f'Mida self.aligned_img_shape_rc: {self.aligned_img_shape_rc}')
        #[1649, 1673]
        self.reference_img_idx = rigid_registrar.reference_img_idx

        #Codi meu
        imatge_a_reg_path = os.path.join(self.mask_dir_rigid_serial_rigid, "output_M_inv.png")
        imatge_a_reg = cv2.imread(imatge_a_reg_path)
        imatge_a_reg_path_2 = os.path.join(self.mask_dir_rigid_serial_rigid, "output_M_inv_2.png")
        imatge_a_reg_2 = cv2.imread(imatge_a_reg_path_2)
        imatge_a_reg_path_3 = os.path.join(self.mask_dir_rigid_serial_rigid, "output_M_inv_3.png")
        imatge_a_reg_3 = cv2.imread(imatge_a_reg_path_3)
        #

        ref_slide = self.slide_dict[valtils.get_name(rigid_registrar.reference_img_f)]
        self.reference_img_f = ref_slide.src_f
        #Codi meu
        #print(f"Nom de la imatge de referència: {self.reference_img_f}")
        #

        self.create_crop_masks(rigid_registrar)
        overlap_mask, overlap_mask_bbox_xywh = self.get_crop_mask(self.crop)

        #Codi meu
        #El que fa la línia de dalt és accedir al diccionari mask_dict amb la clau self.crop i retorna la màscara i la bbox d'aquesta, per això 
        # s'iguala overlap_mask, overlap_mask_bbox_xywh al resultat de lo que retorna la funció. Tot seguit es pot comprovar quina de les màscares utilitza
        #ref_name = self.name_dict[self.reference_img_f]
        #prova = "_get_crop_mask"
        #outline_f_out = os.path.join(self.mask_dir, f'{ref_name}{prova}.png')
        #warp_tools.save_img(outline_f_out, overlap_mask)

        #OBS:Retorna la màscara de self.get_ref_img_mask
        #Obs: crop és un string que indica el mètode que es fa servir per fer el crop
        #
        

        overlap_mask_bbox_xywh = overlap_mask_bbox_xywh.astype(int)
        #Converteix els valors de la bbox_xywh de floats o lo que sigui amb que està guardat a int

        #Codi meu
        #print(f"Mida de la overlap bounding box: {overlap_mask_bbox_xywh}")
        #
        # Create original overlap image #
        self.original_overlap_img = self.create_original_composite_img(rigid_registrar)

        pathlib.Path(self.overlap_dir).mkdir(exist_ok=True, parents=True)
        original_overlap_img_fout = os.path.join(self.overlap_dir, self.name + "_original_overlap.png")
        warp_tools.save_img(original_overlap_img_fout,  self.original_overlap_img, thumbnail_size=self.thumbnail_size)

        pathlib.Path(self.reg_dst_dir).mkdir(exist_ok=  True, parents=  True)
        # Update attributes in slide_obj #
        n_digits = len(str(rigid_registrar.size))
        for slide_reg_obj in rigid_registrar.img_obj_list:
            slide_obj = self.slide_dict[slide_reg_obj.name]
            slide_obj.M = slide_reg_obj.M
            slide_obj.stack_idx = slide_reg_obj.stack_idx
            slide_obj.reg_img_shape_rc = slide_reg_obj.registered_img.shape
            #print(f'Mida de la slide_reg_obj.registered_img.shape: {slide_reg_obj.registered_img.shape}')
            #És [1649, 1673]
            slide_obj.rigid_reg_img_f = os.path.join(self.reg_dst_dir,
                                                     str.zfill(str(slide_obj.stack_idx), n_digits) + "_" + slide_obj.name + ".png")
            #Codi meu
            slide_obj.save_M_field()
            #  
            if slide_obj.image.ndim > 2:
                # Won't know if single channel image is processed RGB (bight bg) or IF channel (dark bg)
                slide_obj.get_bg_color_px_pos()

            if slide_reg_obj.stack_idx == self.reference_img_idx:
                continue

            fixed_slide = self.slide_dict[slide_reg_obj.fixed_obj.name]
            slide_obj.fixed_slide = fixed_slide

            match_dict = slide_reg_obj.match_dict[slide_reg_obj.fixed_obj]
            slide_obj.xy_matched_to_prev = match_dict.matched_kp1_xy
            slide_obj.xy_in_prev = match_dict.matched_kp2_xy

            # Get points in overlap box #
            prev_kp_warped_for_bbox_test = warp_tools.warp_xy(slide_obj.xy_in_prev, M=slide_obj.M)
            _, prev_kp_in_bbox_idx = \
                warp_tools.get_pts_in_bbox(prev_kp_warped_for_bbox_test, overlap_mask_bbox_xywh)

            current_kp_warped_for_bbox_test = \
                warp_tools.warp_xy(slide_obj.xy_matched_to_prev, M=slide_obj.M)

            _, current_kp_in_bbox_idx = \
                warp_tools.get_pts_in_bbox(current_kp_warped_for_bbox_test, overlap_mask_bbox_xywh)

            matched_kp_in_bbox = np.intersect1d(prev_kp_in_bbox_idx, current_kp_in_bbox_idx)
            slide_obj.xy_matched_to_prev_in_bbox =  slide_obj.xy_matched_to_prev[matched_kp_in_bbox]
            slide_obj.xy_in_prev_in_bbox = slide_obj.xy_in_prev[matched_kp_in_bbox]

        if self.denoise_rigid:
            #Entra al denoise
            # Processed image may have been denoised for rigid registration. Replace with unblurred image
            for img_obj in rigid_registrar.img_obj_list:
                matching_slide = self.slide_dict[img_obj.name]
                reg_img = matching_slide.warp_img(matching_slide.processed_img, non_rigid=False, crop=False)
                img_obj.registered_img = reg_img
                img_obj.image = matching_slide.processed_img

        rigid_img_list = [img_obj.registered_img for img_obj in rigid_registrar.img_obj_list]
        self.rigid_overlap_img = self.draw_overlap_img(rigid_img_list)
        #Codi meu
        rigid_overlap_img_path = os.path.join(self.mask_dir_rigid_registration, "Prova_rigid_overlap_img.png")
        warp_tools.save_img(rigid_overlap_img_path, self.rigid_overlap_img)
        #
        self.rigid_overlap_img = warp_tools.crop_img(self.rigid_overlap_img, overlap_mask_bbox_xywh)

        #Codi meu

        #No carrega bé la màscara de output_M_inv pq la Mascara teixit_HE_retallada no està girada, en canvi la output_M_inv.png si que està bé
        imatge_crop =  warp_tools.crop_img(imatge_a_reg, overlap_mask_bbox_xywh)
        imatge_crop_path = os.path.join(self.mask_dir_rigid_registration, "Màscara_teixit_HE_retallada.png")
        warp_tools.save_img(imatge_crop_path, imatge_crop) #, thumbnail_size=self.thumbnail_size)


        imatge_crop_2 =  warp_tools.crop_img(imatge_a_reg_2, overlap_mask_bbox_xywh)
        #imatge_crop_2 = 255 - imatge_crop_2_1
        imatge_crop_path_2 = os.path.join(self.mask_dir_rigid_registration, "Imatge_teixit_HE_retallada_rigid_registered.png")
        warp_tools.save_img(imatge_crop_path_2, imatge_crop_2) #, thumbnail_size=self.thumbnail_size)

        imatge_crop_3 =  warp_tools.crop_img(imatge_a_reg_3, overlap_mask_bbox_xywh)
        imatge_crop_path_3 = os.path.join(self.mask_dir_rigid_registration, "HE_etiquetes_registrada.png")
        warp_tools.save_img(imatge_crop_path_3, imatge_crop_3) #, thumbnail_size=self.thumbnail_size)
        #



        #Codi meu per poder calcular la IoU
        if self.create_masks:
            non_rigid_mask = self._create_mask_from_processed()
        else:
            non_rigid_mask = self._create_non_rigid_reg_mask_from_bbox()
        #

        #Codi meu
        mascara_bbox_non_rig = warp_tools.crop_img(non_rigid_mask, overlap_mask_bbox_xywh)
        bbox_crop_path = os.path.join(self.mask_dir_rigid_registration, "_non_rigid_bbox_retallat.png")
        warp_tools.save_img(bbox_crop_path, mascara_bbox_non_rig) #, thumbnail_size=self.thumbnail_size)
        # Guarda bé la màscara de 1063x1149 amb la bbox 

        rigid_overlap_img_fout = os.path.join(self.overlap_dir, self.name + "_rigid_overlap.png")
        warp_tools.save_img(rigid_overlap_img_fout, self.rigid_overlap_img, thumbnail_size=self.thumbnail_size)



        # Overwrite black and white processed images #
        for slide_name, slide_obj in self.slide_dict.items():
            slide_reg_obj = rigid_registrar.img_obj_dict[slide_name]
            if not slide_obj.is_rgb:
                img_to_warp = slide_reg_obj.image
            else:
                img_to_warp = slide_obj.image
            img_to_warp = warp_tools.resize_img(img_to_warp, slide_obj.processed_img_shape_rc)
            warped_img = slide_obj.warp_img(img_to_warp, non_rigid=False, crop=self.crop)
            warp_tools.save_img(slide_obj.rigid_reg_img_f, warped_img.astype(np.uint8), thumbnail_size=self.thumbnail_size)

            # Replace processed image with a thumbnail #
            warp_tools.save_img(slide_obj.processed_img_f, slide_reg_obj.image, thumbnail_size=self.thumbnail_size)
        
        #Codi meu
        self.iou_rigid = self.compute_iou_from_masks(0)
        self.corr_rigid = self.compute_corr_from_masks(0)
        

        #Codi meu
        he_mask_path = os.path.join(self.mask_dir_rigid_registration, "Màscara_teixit_HE_retallada.png")
        mask1 = imread(he_mask_path, as_gray=True)
        ihc_mask_path = os.path.join(self.mask_dir_rigid_registration, "Màscara_teixit_IHC_original.png")
        mask2 = imread(ihc_mask_path, as_gray=True)
        rigid_mask_overlap = self.draw_overlap_binary_masks(mask1, mask2)

        resultat_path = os.path.join(self.overlap_dir, "Prova_overlap_rigid.png")
        warp_tools.save_img(resultat_path, rigid_mask_overlap)  

        #Codi meu
        #for slide_reg_obj in rigid_registrar.img_obj_list:


        return rigid_registrar





    def non_rigid_register(self, rigid_registrar, processor_dict):

        """Non-rigidly register slides

        Non-rigidly register slides after performing rigid registration.
        Also saves thumbnails of non-rigidly registered images and deformation
        fields.

        Parameters
        ----------
        rigid_registrar : SerialRigidRegistrar
            SerialRigidRegistrar object that performed the rigid registration.

        processor_dict : dict
            Each key should be the filename of the image, and the value either a subclassed
            preprocessing.ImageProcessor, or a list, where the 1st element is the processor,
            and the second element a dictionary of keyword arguments passed to the processor.
            If `None`, then a default processor will be used for each image based on
            the inferred modality.
        Returns
        -------
        non_rigid_registrar : SerialNonRigidRegistrar
            SerialNonRigidRegistrar object that performed serial
            non-rigid registration.

        """

        ref_slide = self.get_ref_slide()
        
        #Codi meu
        #name_ref_slide = ref_slide.name
        #print(f"Agafa la imatge de referència: {name_ref_slide}")
        #
        self.create_non_rigid_reg_mask() #Crea la màscara de la imatge d'IHC, amb fons blanc i teixit i la bbox dibuixada al voltant de la zona a registrar
        non_rigid_reg_mask = ref_slide.non_rigid_reg_mask
        cropped_mask_shape_rc = warp_tools.xy2bbox(warp_tools.mask2xy(non_rigid_reg_mask))[2:][::-1]
        #Codi meu
        #print(f'Mida cropped_mask: {cropped_mask_shape_rc}')
        #Mida (302, 304)

        nr_on_scaled_img = self.max_processed_image_dim_px != self.max_non_rigid_registration_dim_px or \
            (non_rigid_reg_mask is not None and np.any(cropped_mask_shape_rc != ref_slide.reg_img_shape_rc))
        #Al principi es compara les mides de les imatges self.max_processed_image_dim_px que és la mida de les imatges utilitzades pel rigid_reg
        #i es compara amb max_non_rigid_registration_dim_px que és la mida de les imatges utilitzades pel non_rigid_reg. Si els dos valors són iguals implica que 
        #no cal fer cap reescalat i que les imatges que es faran servir pel non_rigid tenen la mateixa resolució que les del rigid. Pel contrari, si són diferents
        #voldrà dir que pel non_rigid fem servir una resolució més gran que pel rígid.
        #nr_on_scaled_img, és un boolean que indica si s'ha de fer el registre no rígid a una imatge escalada (No la original)

        using_tiler = False
        img_specific_args = {}
        if nr_on_scaled_img:
            #Codi meu
            #print(f'Entra al nr_on_scaled_img')
            #Entra aquí
            # Use higher resolution and/or roi for non-rigid
            nr_reg_src, max_img_dim, non_rigid_reg_mask, full_out_shape_rc, mask_bbox_xywh, using_tiler = \
                self.prep_images_for_large_non_rigid_registration(max_img_dim=self.max_non_rigid_registration_dim_px,
                                                                  processor_dict=processor_dict,
                                                                  mask=non_rigid_reg_mask)
            #print(f'Processor_dict: {processor_dict}')

            self._non_rigid_bbox = mask_bbox_xywh
            self.max_non_rigid_registration_dim_px = max_img_dim

            if using_tiler:
                #Codi meu
                #print(f'Entra al using tiler')
                #No entra
                non_rigid_registrar_cls, img_specific_args = self.get_nr_tiling_params(self.non_rigid_reg_kwargs[NON_RIGID_REG_CLASS_KEY],
                                                                                       processor_dict=processor_dict,
                                                                                       img_specific_args=None,
                                                                                       tile_wh=DEFAULT_NR_TILE_WH)

                # Update args to use tiled non-rigid registrar
                self.non_rigid_reg_kwargs[NON_RIGID_REG_CLASS_KEY] = non_rigid_registrar_cls

        else:
            nr_reg_src = rigid_registrar
            full_out_shape_rc = ref_slide.reg_img_shape_rc


        self._full_displacement_shape_rc = full_out_shape_rc #[10182, 10036]
        non_rigid_registrar = serial_non_rigid.register_images(src=nr_reg_src, mask_dir_non_rigid_registration = self.mask_dir_non_rigid_registration,
                                                               mask_dir_non_rigid_serial_non_rigid = self.mask_dir_non_rigid_serial_non_rigid, 
                                                               mask_dir_non_rigid_non_rigid_registrars=self.mask_dir_non_rigid_non_rigid_registrars,
                                                               align_to_reference=self.align_to_reference,
                                                               img_params = img_specific_args,
                                                               **self.non_rigid_reg_kwargs)
        #Codi meu: He afegit la variable mask_dir al codi serial_non_rigid.register_images i li passo self.mask_dir pq pugui accedir al directori per fer debugging i
        #si en un futur cal guardar màscares de les mètriques IoU o correlation
        
        self.end_non_rigid_time = time()

        for d in  [self.non_rigid_dst_dir, self.deformation_field_dir]:
            pathlib.Path(d).mkdir(exist_ok=True, parents=True)
        self.non_rigid_registrar = non_rigid_registrar

        # Clean up displacements and expand if mask was used
        for nr_name, nr_obj in non_rigid_registrar.non_rigid_obj_dict.items():
            if nr_on_scaled_img:
                #Entra

                # If a mask was used, the displacement fields will be smaller
                # So need to insert them in the full image
                bk_dxdy = self.pad_displacement(nr_obj.bk_dxdy, full_out_shape_rc, mask_bbox_xywh)
                fwd_dxdy = self.pad_displacement(nr_obj.fwd_dxdy, full_out_shape_rc, mask_bbox_xywh)
            else:
                bk_dxdy = nr_obj.bk_dxdy
                fwd_dxdy = nr_obj.fwd_dxdy

            nr_obj.bk_dxdy = bk_dxdy
            nr_obj.fwd_dxdy = fwd_dxdy

        # Draw overlap image #
        overlap_mask, overlap_mask_bbox_xywh = self.get_crop_mask(self.crop)
        overlap_mask_bbox_xywh = overlap_mask_bbox_xywh.astype(int)

        if not nr_on_scaled_img:
            non_rigid_img_list = [nr_img_obj.registered_img for nr_img_obj in non_rigid_registrar.non_rigid_obj_list]
            #print(f'Entra non_rigid 1') No entra

        else:
            non_rigid_img_list = []

            for i, o in enumerate(rigid_registrar.img_obj_list):
                # Crea la imatge registrada i afegeix-la a la llista
                result = warp_tools.warp_img(img=o.image,
                                                    M=o.M,
                                                    bk_dxdy= non_rigid_registrar.non_rigid_obj_dict[o.name].bk_dxdy,
                                                    out_shape_rc=o.registered_img.shape[0:2],
                                                    transformation_src_shape_rc=o.image.shape[0:2],
                                                    transformation_dst_shape_rc=o.registered_img.shape[0:2])
                non_rigid_img_list.append(result)

                #Codi meu
                if i == 0:
                    #path_1 = os.path.join(self.mask_dir_non_rigid_registration, "Imatge_non_rigid_img_list.png")
                    #warp_tools.save_img(path_1, o.image)

                    # Aplica warp_img de forma independent per desar (opcional)
                    warp_result = warp_tools.warp_img(
                        img=self.img_etiquetes_he,
                        M=o.M,
                        bk_dxdy=non_rigid_registrar.non_rigid_obj_dict[o.name].bk_dxdy,
                        out_shape_rc=o.registered_img.shape[0:2],
                        transformation_src_shape_rc=o.image.shape[0:2],
                        transformation_dst_shape_rc=o.registered_img.shape[0:2]
                    )

                    # Si vols desar aquesta imatge deformada també:
                    path_2 = os.path.join(self.mask_dir_non_rigid_registration, "He_etiquetes_non_registered_mida_g.png")
                    warp_tools.save_img(path_2, warp_result)

                i += 1
                #
        """
        else:
            iter = 0
            non_rigid_img_list = [warp_tools.warp_img(img=o.image,
                                                    M=o.M,
                                                    bk_dxdy= non_rigid_registrar.non_rigid_obj_dict[o.name].bk_dxdy,
                                                    out_shape_rc=o.registered_img.shape[0:2],
                                                    transformation_src_shape_rc=o.image.shape[0:2],
                                                    transformation_dst_shape_rc=o.registered_img.shape[0:2])
                                                for o in rigid_registrar.img_obj_list]
        """



        self.non_rigid_overlap_img  = self.draw_overlap_img(non_rigid_img_list)
        self.non_rigid_overlap_img = warp_tools.crop_img(self.non_rigid_overlap_img, overlap_mask_bbox_xywh)

        #Codi meu
        warp_result_2 = warp_tools.crop_img(warp_result, overlap_mask_bbox_xywh)
        path_he_etiquetes = os.path.join(self.mask_dir_non_rigid_registration, "He_etiquetes_non_registered_mida_Valis_nivell_8.png")
        warp_tools.save_img(path_he_etiquetes, warp_result_2)

        path_1 = os.path.join(self.mask_dir_rigid_registration , "_non_rigid_bbox_retallat.png")
        bbox_img = imread(path_1, as_gray=True)


        # Posa a 0 només on la màscara és negra
        warp_result_2[bbox_img == 0] = 0
        path_he_etiquetes_2 = os.path.join(self.mask_dir_non_rigid_registration, "He_etiquetes_non_registered_mida_Valis_nivell_8_bbox_aplicada.png")
        warp_tools.save_img(path_he_etiquetes_2, warp_result_2)
        self.preparar_imatge_HE_etiquetes_inversa(self.path_IHC_original, warp_result_2)

        #print(f'Mida overlap_mask_bbox_xywh: {overlap_mask_bbox_xywh}')
        #La mida de overlap_mask_bbox_xywh és [1063, 1149]


        overlap_img_fout = os.path.join(self.overlap_dir, self.name + "_non_rigid_overlap.png")
        warp_tools.save_img(overlap_img_fout, self.non_rigid_overlap_img, thumbnail_size=self.thumbnail_size)

        n_digits = len(str(self.size))
        for slide_name, slide_obj in self.slide_dict.items():
            img_save_id = str.zfill(str(slide_obj.stack_idx), n_digits)
            slide_nr_reg_obj = non_rigid_registrar.non_rigid_obj_dict[slide_name]
            #Codi meu
            #slide_obj.bk_dxdy = slide_nr_reg_obj.bk_dxdy
            #slide_obj.fwd_dxdy = slide_nr_reg_obj.fwd_dxdy
            #

            if not using_tiler:
                slide_obj.bk_dxdy = slide_nr_reg_obj.bk_dxdy
                slide_obj.fwd_dxdy = slide_nr_reg_obj.fwd_dxdy
                #Codi meu
                #slide_obj.save_M_field()
                #
            else:
                # save displacements as images
                pathlib.Path(self.displacements_dir).mkdir(exist_ok=True, parents=True)
                slide_obj.stored_dxdy = True
                bk_dxdy_f, fwd_dxdy_f = slide_obj.get_displacement_f()
                slide_obj._bk_dxdy_f = bk_dxdy_f
                slide_obj._fwd_dxdy_f = fwd_dxdy_f
                # Save space by only writing the necessary areas. Most displacements may be 0
                cropped_bk_dxdy = slide_nr_reg_obj.bk_dxdy.extract_area(*mask_bbox_xywh)
                cropped_fwd_dxdy = slide_nr_reg_obj.fwd_dxdy.extract_area(*mask_bbox_xywh)

                cropped_bk_dxdy.cast("float").tiffsave(slide_obj._bk_dxdy_f, compression="lzw", lossless=True, tile=True, bigtiff=True)
                cropped_fwd_dxdy.cast("float").tiffsave(slide_obj._fwd_dxdy_f, compression="lzw", lossless=True, tile=True, bigtiff=True)

            slide_obj.nr_rigid_reg_img_f = os.path.join(self.non_rigid_dst_dir, img_save_id + "_" + slide_obj.name + ".png")

            if not slide_obj.is_rgb:
                img_to_warp = rigid_registrar.img_obj_dict[slide_name].image
            else:
                img_to_warp = slide_obj.image
            img_to_warp = warp_tools.resize_img(img_to_warp, slide_obj.processed_img_shape_rc)
            warped_img = slide_obj.warp_img(img_to_warp, non_rigid=True, crop=self.crop)
            warp_tools.save_img(slide_obj.nr_rigid_reg_img_f, warped_img, thumbnail_size=self.thumbnail_size)

            # Draw displacements on image actually used in non-rigid. Might be higher resolution
            if not isinstance(slide_nr_reg_obj.bk_dxdy, pyvips.Image):
                draw_dxdy = np.dstack(slide_nr_reg_obj.bk_dxdy)
            else:
                #pyvips
                draw_dxdy = slide_nr_reg_obj.bk_dxdy

            if nr_on_scaled_img:
                draw_dxdy = warp_tools.crop_img(draw_dxdy, self._non_rigid_bbox)

            dxdy_shape = warp_tools.get_shape(draw_dxdy)
            thumbnail_scaling = np.min(self.thumbnail_size/np.array(dxdy_shape[0:2]))
            thumbnail_bk_dxdy = self.create_thumbnail(draw_dxdy)
            thumbnail_bk_dxdy *= float(thumbnail_scaling)

            if isinstance(thumbnail_bk_dxdy, pyvips.Image):
                thumbnail_bk_dxdy = warp_tools.vips2numpy(thumbnail_bk_dxdy)

            draw_img = warp_tools.resize_img(slide_nr_reg_obj.registered_img, thumbnail_bk_dxdy[..., 0].shape)
            if isinstance(draw_img, pyvips.Image):
                draw_img = warp_tools.vips2numpy(draw_img)

            draw_img = exposure.rescale_intensity(draw_img, out_range=(0, 255))

            if draw_img.ndim == 2:
                draw_img = np.dstack([draw_img] * 3)

            thumbanil_deform_grid = viz.color_displacement_tri_grid(bk_dx=thumbnail_bk_dxdy[..., 0],
                                                                    bk_dy=thumbnail_bk_dxdy[..., 1],
                                                                    img=draw_img,
                                                                    n_grid_pts=25)

            deform_img_f = os.path.join(self.deformation_field_dir, img_save_id + "_" + slide_obj.name + ".png")
            warp_tools.save_img(deform_img_f, thumbanil_deform_grid, thumbnail_size=self.thumbnail_size)
        
        #Codi meu
        self.iou_non_rigid = self.compute_iou_from_masks(1)
        self.corr_non_rigid = self.compute_corr_from_masks(1)

        #Codi meu
        i = 0
        resultat_path = os.path.join(self.overlap_dir, "Prova_overlap_original.png")
        he_mask_path = os.path.join(self.mask_dir_rigid_registration, "Màscara_teixit_HE_original.png")
        mask1 = imread(he_mask_path, as_gray=True)
        ihc_mask_path = os.path.join(self.mask_dir_rigid_registration, "Màscara_teixit_IHC_original.png")
        mask2 = imread(ihc_mask_path, as_gray=True)
        original_mask_overlap = self.draw_overlap_binary_masks(mask1, mask2)
        warp_tools.save_img(resultat_path, original_mask_overlap) 

        for o in rigid_registrar.img_obj_list:
            if i == 0:
                mask11_path = os.path.join(self.mask_dir_non_rigid_registration, "mask1_1.png")
                warp_tools.save_img(mask11_path, o.image)
                mask1 = warp_tools.warp_img(img=mask1,
                                                    M=o.M,
                                                    bk_dxdy= non_rigid_registrar.non_rigid_obj_dict[o.name].bk_dxdy,
                                                    out_shape_rc=o.registered_img.shape[0:2],
                                                    transformation_src_shape_rc=o.image.shape[0:2],
                                                    transformation_dst_shape_rc=o.registered_img.shape[0:2])
                mask1_path = os.path.join(self.mask_dir_non_rigid_registration, "mask1.png")
                warp_tools.save_img(mask1_path, mask1)
            if i == 1:
                mask21_path = os.path.join(self.mask_dir_non_rigid_registration, "mask2_1.png")
                warp_tools.save_img(mask21_path, o.image)
                mask2 = warp_tools.warp_img(img=mask2,
                                                    M=o.M,
                                                    bk_dxdy= non_rigid_registrar.non_rigid_obj_dict[o.name].bk_dxdy,
                                                    out_shape_rc=o.registered_img.shape[0:2],
                                                    transformation_src_shape_rc=o.image.shape[0:2],
                                                    transformation_dst_shape_rc=o.registered_img.shape[0:2])
                mask2_path = os.path.join(self.mask_dir_non_rigid_registration, "mask2.png")
                warp_tools.save_img(mask2_path, mask2)
            i = i+1
              
                                                      
        non_rigid_mask_overlap = self.draw_overlap_binary_masks(mask1, mask2)
        non_rigid_mask_overlap = warp_tools.crop_img(non_rigid_mask_overlap, overlap_mask_bbox_xywh)
        resultat_path = os.path.join(self.overlap_dir, "Prova_overlap_non_rigid.png")
        warp_tools.save_img(resultat_path, non_rigid_mask_overlap)  
        #

        return non_rigid_registrar

    def measure_error(self):
        """Measure registration error

        Error is measured as the distance between matched features
        after registration.

        Returns
        -------
        summary_df : Dataframe
            `summary_df` contains various information about the registration.

            The "from" column is the name of the image, while the "to" column
            name of the image it was aligned to. "from" is analagous to "moving"
            or "current", while "to" is analgous to "fixed" or "previous".

            Columns begining with "original" refer to error measurements of the
            unregistered images. Those beginning with "rigid" or "non_rigid" refer
            to measurements related to rigid or non-rigid registration, respectively.

            Columns beginning with "mean" are averages of error measurements. In
            the case of errors based on feature distances (i.e. those ending in "D"),
            the mean is weighted by the number of feature matches between "from" and "to".

            Columns endining in "D" indicate the median distance between matched
            features in "from" and "to".

            Columns ending in "rTRE" indicate the target registration error between
            "from" and "to".

            Columns ending in "mattesMI" contain measurements of the Mattes mutual
            information between "from" and "to".

            "processed_img_shape" indicates the shape (row, column) of the processed
            image actually used to conduct the registration

            "shape" is the shape of the slide at full resolution

            "aligned_shape" is the shape of the registered full resolution slide

            "physical_units" are the names of the pixels physcial unit, e.g. u'\u00B5m'

            "resolution" is the physical unit per pixel

            "name" is the name assigned to the Valis instance

            "rigid_time_minutes" is the total number of minutes it took
            to convert the images and then rigidly align them.

            "non_rigid_time_minutes" is the total number of minutes it took
            to convert the images, and then perform rigid -> non-rigid registration.

        """

        path_list = [None] * (self.size)
        all_og_d = [None] * (self.size)
        all_og_tre = [None] * (self.size)

        all_rigid_d = [None] * (self.size)
        all_rigid_tre = [None] * (self.size)

        all_nr_d = [None] * (self.size)
        all_nr_tre = [None] * (self.size)

        all_n = [None] * (self.size)
        from_list = [None] * (self.size)
        to_list = [None] * (self.size)
        shape_list = [None] * (self.size)
        processed_img_shape_list = [None] * (self.size)
        unit_list = [None] * (self.size)
        resolution_list = [None] * (self.size)

        slide_obj_list = list(self.slide_dict.values())
        outshape = slide_obj_list[0].aligned_slide_shape_rc

        ref_slide = self.get_ref_slide()
        ref_diagonal = np.sqrt(np.sum(np.power(ref_slide.processed_img_shape_rc, 2)))

        measure_idx = []
        for slide_obj in tqdm.tqdm(self.slide_dict.values(), desc=MEASURE_MSG, unit="image"):
            i = slide_obj.stack_idx
            slide_name = slide_obj.name
            #Codi meu
            #print(f"Mesurant l'error de {slide_name}")
            #
            shape_list[i] = tuple(slide_obj.slide_shape_rc)
            processed_img_shape_list[i] = tuple(slide_obj.processed_img_shape_rc)
            unit_list[i] = slide_obj.units
            resolution_list[i] = slide_obj.resolution
            from_list[i] = slide_name
            path_list[i] = slide_obj.src_f

            if slide_obj.name == ref_slide.name or slide_obj.is_empty:
                continue

            measure_idx.append(i)
            prev_slide_obj = slide_obj.fixed_slide
            to_list[i] = prev_slide_obj.name

            img_T = warp_tools.get_padding_matrix(slide_obj.processed_img_shape_rc,
                                                  slide_obj.reg_img_shape_rc)

            prev_T = warp_tools.get_padding_matrix(prev_slide_obj.processed_img_shape_rc,
                                                   prev_slide_obj.reg_img_shape_rc)


            prev_kp_in_slide = prev_slide_obj.warp_xy(slide_obj.xy_in_prev,
                                                     M=prev_T,
                                                     pt_level= prev_slide_obj.processed_img_shape_rc,
                                                     non_rigid=False)

            current_kp_in_slide = slide_obj.warp_xy(slide_obj.xy_matched_to_prev,
                                                    M=img_T,
                                                    pt_level= slide_obj.processed_img_shape_rc,
                                                    non_rigid=False)

            og_d = warp_tools.calc_d(prev_kp_in_slide, current_kp_in_slide)

            og_rtre = og_d/ref_diagonal
            median_og_tre = np.median(og_rtre)
            og_d *= slide_obj.resolution
            median_d_og = np.median(og_d)

            all_og_d[i] = median_d_og
            all_og_tre[i] = median_og_tre


            prev_warped_rigid = prev_slide_obj.warp_xy(slide_obj.xy_in_prev,
                                                       M=prev_slide_obj.M,
                                                       pt_level= prev_slide_obj.processed_img_shape_rc,
                                                       non_rigid=False)

            current_warped_rigid = slide_obj.warp_xy(slide_obj.xy_matched_to_prev,
                                                     M=slide_obj.M,
                                                     pt_level= slide_obj.processed_img_shape_rc,
                                                     non_rigid=False)


            rigid_d = warp_tools.calc_d(prev_warped_rigid, current_warped_rigid)
            rtre = rigid_d/ref_diagonal
            median_rigid_tre = np.median(rtre)
            rigid_d *= slide_obj.resolution
            median_d_rigid = np.median(rigid_d)

            all_rigid_d[i] = median_d_rigid
            all_n[i] = len(rigid_d)
            all_rigid_tre[i] = median_rigid_tre

            if slide_obj.bk_dxdy is not None:


                prev_warped_nr = prev_slide_obj.warp_xy(slide_obj.xy_in_prev,
                                                        M=prev_slide_obj.M,
                                                        pt_level= prev_slide_obj.processed_img_shape_rc,
                                                        non_rigid=True)

                current_warped_nr = slide_obj.warp_xy(slide_obj.xy_matched_to_prev,
                                                      M=slide_obj.M,
                                                      pt_level= slide_obj.processed_img_shape_rc,
                                                      non_rigid=True)

                nr_d =  warp_tools.calc_d(prev_warped_nr, current_warped_nr)
                nrtre = nr_d/ref_diagonal
                mean_nr_tre = np.median(nrtre)

                nr_d *= slide_obj.resolution
                median_d_nr = np.median(nr_d)
                all_nr_d[i] = median_d_nr
                all_nr_tre[i] = mean_nr_tre

        weights = np.array(all_n)[measure_idx]
        mean_og_d = np.average(np.array(all_og_d)[measure_idx], weights=weights)
        median_og_tre = np.average(np.array(all_og_tre)[measure_idx], weights=weights)

        mean_rigid_d = np.average(np.array(all_rigid_d)[measure_idx], weights=weights)
        median_rigid_tre = np.average(np.array(all_rigid_tre)[measure_idx], weights=weights)

        rigid_min = (self.end_rigid_time - self.start_time)/60

        self.summary_df = pd.DataFrame({
            "filename": path_list,
            "from":from_list,
            "to": to_list,
            "original_D": all_og_d,
            "original_rTRE": all_og_tre,
            "rigid_D": all_rigid_d,
            "rigid_rTRE": all_rigid_tre,
            "non_rigid_D": all_nr_d,
            "non_rigid_rTRE": all_nr_tre,
            "processed_img_shape": processed_img_shape_list,
            "shape": shape_list,
            "aligned_shape": [tuple(outshape)]*self.size,
            "mean_original_D": [mean_og_d]*self.size,
            "mean_rigid_D": [mean_rigid_d]*self.size,
            "physical_units":unit_list,
            "resolution":resolution_list,
            "name": [self.name]*self.size,
            "rigid_time_minutes" : [rigid_min]*self.size
        })

        if any([d for d in all_nr_d if d is not None]):
            mean_nr_d = np.average(np.array(all_nr_d)[measure_idx], weights=weights)
            mean_nr_tre = np.average(np.array(all_nr_tre)[measure_idx], weights=weights)
            non_rigid_min = (self.end_non_rigid_time - self.start_time)/60

            self.summary_df["mean_non_rigid_D"] = [mean_nr_d]*self.size
            self.summary_df["non_rigid_time_minutes"] = [non_rigid_min]*self.size

        return self.summary_df

    def register(self, brightfield_processing_cls=DEFAULT_BRIGHTFIELD_CLASS,
                 brightfield_processing_kwargs=DEFAULT_BRIGHTFIELD_PROCESSING_ARGS,
                 if_processing_cls=DEFAULT_FLOURESCENCE_CLASS,
                 if_processing_kwargs=DEFAULT_FLOURESCENCE_PROCESSING_ARGS,
                 processor_dict=None,
                 reader_cls=None,
                 reader_dict=None):

        """Register a collection of images

        This function will convert the slides to images, pre-process and normalize them, and
        then conduct rigid registration. Non-rigid registration will then be performed if the
        `non_rigid_registrar_cls` argument used to initialize the Valis object was not None.

        In addition to the objects returned, the desination directory (i.e. `dst_dir`)
        will contain thumbnails so that one can visualize the results: converted image
        thumbnails will be in "images/"; processed images in "processed/";
        rigidly aligned images in "rigid_registration/"; non-rigidly aligned images in "non_rigid_registration/";
        non-rigid deformation field images (i.e. warped grids colored by the direction and magntidue)
        of the deformation) will be in ""deformation_fields/". The size of these thumbnails
        is determined by the `thumbnail_size` argument used to initialze this object.

        One can get a sense of how well the registration worked by looking
        in the "overlaps/", which shows how the images overlap before
        registration, after rigid registration, and after non-rigid registration. Each image
        is created by coloring an inverted greyscale version of the processed images, and then
        blending those images.

        The "data/" directory will contain a pickled copy of this registrar, which can be
        later be opened (unpickled) and used to warp slides and/or point data.

        "data/" will also contain the `summary_df` saved as a csv file.


        Parameters
        ----------
        brightfield_processing_cls : preprocessing.ImageProcesser
            preprocessing.ImageProcesser used to pre-process brightfield images to make
            them look as similar as possible.

        brightfield_processing_kwargs : dict
            Dictionary of keyward arguments to be passed to `brightfield_processing_cls`

        if_processing_cls : preprocessing.ImageProcesser
            preprocessing.ImageProcesser used to pre-process immunofluorescent images
            to make them look as similar as possible.

        if_processing_kwargs : dict
            Dictionary of keyward arguments to be passed to `if_processing_cls`

        processor_dict : dict, optional
            Each key should be the filename of the image, and the value either a subclassed
            preprocessing.ImageProcessor, or a list, where the 1st element is the processor,
            and the second element a dictionary of keyword arguments passed to the processor.
            If `None`, then a default processor will be used for each image based on
            the inferred modality.

        reader_cls : SlideReader, optional
            Uninstantiated SlideReader class that will convert
            the slide to an image, and also collect metadata. If None (the default),
            the appropriate SlideReader will be found by `slide_io.get_slide_reader`.
            This option is provided in case the slides cannot be opened by a current
            SlideReader class. In this case, the user should create a subclass of
            SlideReader. See slide_io.SlideReader for details.

        reader_dict: dict, optional
            Dictionary specifying which readers to use for individual images. The
            keys should be the image's filename, and the values the instantiated slide_io.SlideReader
            to use to read that file. Valis will try to find an appropritate reader
            for any omitted files, or will use `reader_cls` as the default.

        Returns
        -------
        rigid_registrar : SerialRigidRegistrar
            SerialRigidRegistrar object that performed the rigid registration.
            This object can be pickled if so desired

        non_rigid_registrar : SerialNonRigidRegistrar
            SerialNonRigidRegistrar object that performed serial
            non-rigid registration. This object can be pickled if so desired.

        summary_df : Dataframe
            `summary_df` contains various information about the registration.

            The "from" column is the name of the image, while the "to" column
            name of the image it was aligned to. "from" is analagous to "moving"
            or "current", while "to" is analgous to "fixed" or "previous".

            Columns begining with "original" refer to error measurements of the
            unregistered images. Those beginning with "rigid" or "non_rigid" refer
            to measurements related to rigid or non-rigid registration, respectively.

            Columns beginning with "mean" are averages of error measurements. In
            the case of errors based on feature distances (i.e. those ending in "D"),
            the mean is weighted by the number of feature matches between "from" and "to".

            Columns endining in "D" indicate the median distance between matched
            features in "from" and "to".

            Columns ending in "TRE" indicate the target registration error between
            "from" and "to".

            Columns ending in "mattesMI" contain measurements of the Mattes mutual
            information between "from" and "to".

            "processed_img_shape" indicates the shape (row, column) of the processed
            image actually used to conduct the registration

            "shape" is the shape of the slide at full resolution

            "aligned_shape" is the shape of the registered full resolution slide

            "physical_units" are the names of the pixels physcial unit, e.g. u'\u00B5m'

            "resolution" is the physical unit per pixel

            "name" is the name assigned to the Valis instance

            "rigid_time_minutes" is the total number of minutes it took
            to convert the images and then rigidly align them.

            "non_rigid_time_minutes" is the total number of minutes it took
            to convert the images, and then perform rigid -> non-rigid registration.

        """

        self.start_time = time()
        try:
            print("\n==== Converting images\n")
            self.convert_imgs(series=self.series, reader_cls=reader_cls, reader_dict=reader_dict)


            print("\n==== Processing images\n")
            slide_processors = self.create_img_processor_dict(brightfield_processing_cls=brightfield_processing_cls,
                                            brightfield_processing_kwargs=brightfield_processing_kwargs,
                                            if_processing_cls=if_processing_cls,
                                            if_processing_kwargs=if_processing_kwargs,
                                            processor_dict=processor_dict)

            self.brightfield_procsseing_fxn_str = brightfield_processing_cls.__name__
            self.if_processing_fxn_str = if_processing_cls.__name__
            self.process_imgs(processor_dict=slide_processors)

            # print("\n==== Rigid registration\n")
            rigid_registrar = self.rigid_register()
            aligned_slide_shape_rc = self.get_aligned_slide_shape(0)
            self.aligned_slide_shape_rc = aligned_slide_shape_rc
            self.iter_order = rigid_registrar.iter_order
            for slide_obj in self.slide_dict.values():
                slide_obj.aligned_slide_shape_rc = aligned_slide_shape_rc

            if self.micro_rigid_registrar_cls is not None:
                print("\n==== Micro-rigid registration\n")
                self.micro_rigid_register()

            if rigid_registrar is False:
                return None, None, None

            if self.non_rigid_registrar_cls is not None:
                print("\n==== Non-rigid registration\n")
                non_rigid_registrar = self.non_rigid_register(rigid_registrar, slide_processors)

            else:
                non_rigid_registrar = None


            self._add_empty_slides()

            print("\n==== Measuring error\n")
            error_df = self.measure_error()
            self.cleanup()

            
            pathlib.Path(self.data_dir).mkdir(exist_ok=True,  parents=True)
            '''
            f_out = os.path.join(self.data_dir, self.name + "_registrar.pickle")
            self.reg_f = f_out
            pickle.dump(self, open(f_out, 'wb'))
            '''

            data_f_out = os.path.join(self.data_dir, self.name + "_summary.csv")
            error_df.to_csv(data_f_out, index=False)

        except Exception as e:
            traceback_msg = traceback.format_exc()
            valtils.print_warning(e, rgb=Fore.RED, traceback_msg=traceback_msg)
            kill_jvm()
            return None, None, None

        iou_rigid = self.iou_rigid
        corr_rigid = self.corr_rigid
        iou_non_rigid = self.iou_non_rigid
        corr_non_rigid = self.corr_non_rigid
        return rigid_registrar, non_rigid_registrar, error_df, iou_rigid, corr_rigid, iou_non_rigid, corr_non_rigid

    def cleanup(self):
        """Remove objects that can't be pickled
        """
        self.rigid_reg_kwargs["feature_detector"] = None
        self.rigid_reg_kwargs["affine_optimizer"] = None
        self.non_rigid_registrar_cls = None
        self.rigid_registrar = None
        self.micro_rigid_registrar_cls = None
        self.non_rigid_registrar = None


    @valtils.deprecated_args(max_non_rigid_registartion_dim_px="max_non_rigid_registration_dim_px")
    def register_micro(self,  brightfield_processing_cls=DEFAULT_BRIGHTFIELD_CLASS,
                 brightfield_processing_kwargs=DEFAULT_BRIGHTFIELD_PROCESSING_ARGS,
                 if_processing_cls=DEFAULT_FLOURESCENCE_CLASS,
                 if_processing_kwargs=DEFAULT_FLOURESCENCE_PROCESSING_ARGS,
                 processor_dict=None,
                 max_non_rigid_registration_dim_px=DEFAULT_MAX_NON_RIGID_REG_SIZE,
                 non_rigid_registrar_cls=DEFAULT_NON_RIGID_CLASS,
                 non_rigid_reg_params=DEFAULT_NON_RIGID_KWARGS,
                 reference_img_f=None, align_to_reference=False, mask=None, tile_wh=DEFAULT_NR_TILE_WH):
        """Improve alingment of microfeatures by performing second non-rigid registration on larger images

        Caclculates additional non-rigid deformations using a larger image

        Parameters
        ----------
        brightfield_processing_cls : preprocessing.ImageProcesser
            preprocessing.ImageProcesser used to pre-process brightfield images to make
            them look as similar as possible.

        brightfield_processing_kwargs : dict
            Dictionary of keyward arguments to be passed to `brightfield_processing_cls`

        if_processing_cls : preprocessing.ImageProcesser
            preprocessing.ImageProcesser used to pre-process immunofluorescent images
            to make them look as similar as possible.

        if_processing_kwargs : dict
            Dictionary of keyward arguments to be passed to `if_processing_cls`

        max_non_rigid_registration_dim_px : int, optional
             Maximum width or height of images used for non-rigid registration.
             If None, then the full sized image will be used. However, this
             may take quite some time to complete.

        reference_img_f : str, optional
            Filename of image that will be treated as the center of the stack.
            If None, the index of the middle image will be the reference, and
            images will be aligned towards it. If provided, images will be
            aligned to this reference.

        align_to_reference : bool, optional
            If `False`, images will be non-rigidly aligned serially towards the
            reference image. If `True`, images will be non-rigidly aligned
            directly to the reference image. If `reference_img_f` is None,
            then the reference image will be the one in the middle of the stack.

        non_rigid_registrar_cls : NonRigidRegistrar, optional
            Uninstantiated NonRigidRegistrar class that will be used to
            calculate the deformation fields between images. See
            the `non_rigid_registrars` module for a desciption of available
            methods. If a desired non-rigid registration method is not available,
            one can be implemented by subclassing.NonRigidRegistrar.

        non_rigid_reg_params: dictionary, optional
            Dictionary containing key, value pairs to be used to initialize
            `non_rigid_registrar_cls`.
            In the case where simple ITK is used by the, params should be
            a SimpleITK.ParameterMap. Note that numeric values nedd to be
            converted to strings. See the NonRigidRegistrar classes in
            `non_rigid_registrars` for the available non-rigid registration
            methods and arguments.

        """


        # Remove empty slides
        for empty_slide_name, empty_slide in self._empty_slides.items():
            del self.slide_dict[empty_slide_name]
            self.size -= 1

        ref_slide = self.get_ref_slide()
        if mask is None:
            if ref_slide.non_rigid_reg_mask is not None:
                mask = ref_slide.non_rigid_reg_mask.copy()

        slide_processors = self.create_img_processor_dict(brightfield_processing_cls=brightfield_processing_cls,
                                brightfield_processing_kwargs=brightfield_processing_kwargs,
                                if_processing_cls=if_processing_cls,
                                if_processing_kwargs=if_processing_kwargs,
                                processor_dict=processor_dict)

        nr_reg_src, max_img_dim, non_rigid_reg_mask, full_out_shape_rc, mask_bbox_xywh, using_tiler = \
            self.prep_images_for_large_non_rigid_registration(max_img_dim=max_non_rigid_registration_dim_px,
                                                              processor_dict=slide_processors,
                                                              updating_non_rigid=True,
                                                              mask=mask)

        img_specific_args = None
        write_dxdy = isinstance(ref_slide.bk_dxdy, pyvips.Image)

        if using_tiler:
            # Have determined that these images will be too big
            msg = (f"Registration would more than {TILER_THRESH_GB} GB if all images opened in memory. "
                    f"Will use NonRigidTileRegistrar to register cooresponding tiles to reduce memory consumption, "
                    f"but this method is experimental")

            valtils.print_warning(msg)

            write_dxdy = True
            non_rigid_registrar_cls, img_specific_args = self.get_nr_tiling_params(non_rigid_registrar_cls,
                                                                                   processor_dict=slide_processors,
                                                                                   img_specific_args=img_specific_args,
                                                                                   tile_wh=tile_wh)

        print("\n==== Performing microregistration\n")
        non_rigid_registrar = serial_non_rigid.register_images(src=nr_reg_src,
                                                               non_rigid_reg_class=non_rigid_registrar_cls,
                                                               non_rigid_reg_params=non_rigid_reg_params,
                                                               reference_img_f=reference_img_f,
                                                               mask=non_rigid_reg_mask,
                                                               align_to_reference=align_to_reference,
                                                               name=self.name,
                                                               img_params=img_specific_args
                                                               )

        pathlib.Path(self.micro_reg_dir).mkdir(exist_ok=True, parents=True)
        out_shape = full_out_shape_rc
        n_digits = len(str(self.size))
        micro_reg_imgs = [None] * self.size

        # Update displacements
        for slide_obj in self.slide_dict.values():

            if slide_obj == ref_slide:
                continue

            nr_obj = non_rigid_registrar.non_rigid_obj_dict[slide_obj.name]
            # Will be combining original and new dxdy as pyvips Images
            if not isinstance(slide_obj.bk_dxdy[0], pyvips.Image):
                vips_current_bk_dxdy = warp_tools.numpy2vips(np.dstack(slide_obj.bk_dxdy)).cast("float")
                vips_current_fwd_dxdy = warp_tools.numpy2vips(np.dstack(slide_obj.fwd_dxdy)).cast("float")
            else:
                vips_current_bk_dxdy = slide_obj.bk_dxdy
                vips_current_fwd_dxdy = slide_obj.fwd_dxdy

            if not isinstance(nr_obj.bk_dxdy, pyvips.Image):
                vips_new_bk_dxdy = warp_tools.numpy2vips(np.dstack(nr_obj.bk_dxdy)).cast("float")
                vips_new_fwd_dxdy = warp_tools.numpy2vips(np.dstack(nr_obj.fwd_dxdy)).cast("float")
            else:
                vips_new_bk_dxdy = nr_obj.bk_dxdy
                vips_new_fwd_dxdy = nr_obj.fwd_dxdy

            if np.any(non_rigid_registrar.shape != full_out_shape_rc):
                # Micro-registration performed on sub-region. Need to put in full image
                vips_new_bk_dxdy = self.pad_displacement(vips_new_bk_dxdy, full_out_shape_rc, mask_bbox_xywh)
                vips_new_fwd_dxdy = self.pad_displacement(vips_new_fwd_dxdy, full_out_shape_rc, mask_bbox_xywh)

            # Scale original dxdy to match scaled shape of new dxdy
            slide_sxy = (np.array(out_shape)/np.array([vips_current_bk_dxdy.height, vips_current_bk_dxdy.width]))[::-1]
            if not np.all(slide_sxy == 1):
                scaled_bk_dx = float(slide_sxy[0])*vips_current_bk_dxdy[0]
                scaled_bk_dy = float(slide_sxy[1])*vips_current_bk_dxdy[1]
                vips_current_bk_dxdy = scaled_bk_dx.bandjoin(scaled_bk_dy)
                vips_current_bk_dxdy = warp_tools.resize_img(vips_current_bk_dxdy, out_shape)

                scaled_fwd_dx = float(slide_sxy[0])*vips_current_fwd_dxdy[0]
                scaled_fwd_dy = float(slide_sxy[1])*vips_current_fwd_dxdy[1]
                vips_current_fwd_dxdy = scaled_fwd_dx.bandjoin(scaled_fwd_dy)
                vips_current_fwd_dxdy = warp_tools.resize_img(vips_current_fwd_dxdy, out_shape)

            vips_updated_bk_dxdy = vips_current_bk_dxdy + vips_new_bk_dxdy
            vips_updated_fwd_dxdy = vips_current_fwd_dxdy + vips_new_fwd_dxdy

            if not write_dxdy:
                # Will save numpy dxdy as Slide attributes
                np_updated_bk_dxdy = warp_tools.vips2numpy(vips_updated_bk_dxdy)
                np_updated_fwd_dxdy = warp_tools.vips2numpy(vips_updated_fwd_dxdy)

                slide_obj.bk_dxdy = np.array([np_updated_bk_dxdy[..., 0], np_updated_bk_dxdy[..., 1]])
                slide_obj.fwd_dxdy = np.array([np_updated_fwd_dxdy[..., 0], np_updated_fwd_dxdy[..., 1]])
            else:
                pathlib.Path(self.displacements_dir).mkdir(exist_ok=True, parents=True)
                slide_obj.stored_dxdy = True

                bk_dxdy_f, fwd_dxdy_f = slide_obj.get_displacement_f()
                slide_obj._bk_dxdy_f = bk_dxdy_f
                slide_obj._fwd_dxdy_f = fwd_dxdy_f

                # Save space by only writing the necessary areas. Most displacements may be 0
                cropped_bk_dxdy = vips_updated_bk_dxdy.extract_area(*mask_bbox_xywh)
                cropped_fwd_dxdy = vips_updated_fwd_dxdy.extract_area(*mask_bbox_xywh)

                if not os.path.exists(slide_obj._bk_dxdy_f):
                    cropped_bk_dxdy.cast("float").tiffsave(slide_obj._bk_dxdy_f, compression="lzw", lossless=True, tile=True, bigtiff=True)

                else:
                    # Don't seem to be able to overwrite directly because also accessing it?
                    disp_dir, temp_bk_f = os.path.split(slide_obj._bk_dxdy_f)
                    full_temp_dx_f = os.path.join(disp_dir, f".temp_{temp_bk_f}")
                    cropped_bk_dxdy.cast("float").tiffsave(full_temp_dx_f, compression="lzw", lossless=True, tile=True, bigtiff=True)
                    os.remove(slide_obj._bk_dxdy_f)
                    os.rename(full_temp_dx_f, slide_obj._bk_dxdy_f)

                if not os.path.exists(slide_obj._fwd_dxdy_f):
                    cropped_fwd_dxdy.cast("float").tiffsave(slide_obj._fwd_dxdy_f, compression="lzw", lossless=True, tile=True, bigtiff=True)
                else:
                    disp_dir, temp_fwd_f = os.path.split(slide_obj._fwd_dxdy_f)
                    full_temp_fwd_f = os.path.join(disp_dir, f".temp_{temp_fwd_f}")
                    cropped_fwd_dxdy.cast("float").tiffsave(full_temp_fwd_f, compression="lzw", lossless=True, tile=True, bigtiff=True)
                    os.remove(slide_obj._fwd_dxdy_f)
                    os.rename(full_temp_fwd_f, slide_obj._fwd_dxdy_f)

        # Update dxdy padding attributes here, in the event that previous displacements were also saved as files
        # Updating these attributes earlier will cause errors
        self._non_rigid_bbox = mask_bbox_xywh
        self._full_displacement_shape_rc = full_out_shape_rc
        for slide_obj in self.slide_dict.values():
            if not slide_obj.is_rgb:
                img_to_warp = slide_obj.processed_img
            else:
                img_to_warp = slide_obj.image

            img_to_warp = warp_tools.resize_img(img_to_warp, slide_obj.processed_img_shape_rc)
            micro_reg_img = slide_obj.warp_img(img_to_warp, non_rigid=True, crop=self.crop)

            img_save_id = str.zfill(str(slide_obj.stack_idx), n_digits)
            micro_fout = os.path.join(self.micro_reg_dir, f"{img_save_id}_{slide_obj.name}.png")
            micro_thumb = self.create_thumbnail(micro_reg_img)
            warp_tools.save_img(micro_fout, micro_thumb)

            processed_micro_reg_img = slide_obj.warp_img(slide_obj.processed_img)
            micro_reg_imgs[slide_obj.stack_idx] = processed_micro_reg_img


        # Add empty slides back and save results
        for empty_slide_name, empty_slide in self._empty_slides.items():
            self.slide_dict[empty_slide_name] = empty_slide
            self.size += 1

        pickle.dump(self, open(self.reg_f, 'wb'))

        micro_overlap = self.draw_overlap_img(micro_reg_imgs)
        self.micro_reg_overlap_img = micro_overlap
        overlap_img_fout = os.path.join(self.overlap_dir, self.name + "_micro_reg.png")
        warp_tools.save_img(overlap_img_fout, micro_overlap, thumbnail_size=self.thumbnail_size)


        print("\n==== Measuring error\n")
        error_df = self.measure_error()
        data_f_out = os.path.join(self.data_dir, self.name + "_summary.csv")
        error_df.to_csv(data_f_out, index=False)

        return non_rigid_registrar, error_df

    def get_aligned_slide_shape(self, level):
        """Get size of aligned images

        Parameters
        ----------
        level : int, float
            If `level` is an integer, then it is assumed that `level` is referring to
            the pyramid level that will be warped.

            If `level` is a float, it is assumed `level` is how much to rescale the
            registered image's size.

        """

        ref_slide = self.get_ref_slide()

        if np.issubdtype(type(level), np.integer):
            n_levels = len(ref_slide.slide_dimensions_wh)
            if level >= n_levels:
                msg = (f"requested to scale transformation for pyramid level {level}, ",
                    f"but the image only has {n_levels} (starting from 0). ",
                    f"Will use level {level-1}, which is the smallest level")
                valtils.print_warning(msg)
                level = level - 1

            slide_shape_rc = ref_slide.slide_dimensions_wh[level][::-1]
            s_rc = (slide_shape_rc/np.array(ref_slide.processed_img_shape_rc))
        else:
            s_rc = level

        aligned_out_shape_rc = np.ceil(np.array(ref_slide.reg_img_shape_rc)*s_rc).astype(int)

        return aligned_out_shape_rc


    @valtils.deprecated_args(perceputally_uniform_channel_colors="colormap")
    def warp_and_save_slides(self, dst_dir, level=0, non_rigid=True,
                             crop=True,
                             colormap=slide_io.CMAP_AUTO,
                             interp_method="bicubic",
                             tile_wh=None, compression="lzw", Q=100, pyramid=True):

        f"""Warp and save all slides

        Each slide will be saved as an ome.tiff. The extension of each file will
        be changed to ome.tiff if it is not already.

        Parameters
        ----------
        dst_dir : str
            Path to were the warped slides will be saved.

        level : int, optional
            Pyramid level to be warped. Default is 0, which means the highest
            resolution image will be warped and saved.

        non_rigid : bool, optional
            Whether or not to conduct non-rigid warping. If False,
            then only a rigid transformation will be applied. Default is True

        crop: bool, str
            How to crop the registered images. If `True`, then the same crop used
            when initializing the `Valis` object will be used. If `False`, the
            image will not be cropped. If "overlap", the warped slide will be
            cropped to include only areas where all images overlapped.
            "reference" crops to the area that overlaps with the reference image,
            defined by `reference_img_f` when initialzing the `Valis object`.

        colormap : list
            List of RGB colors (0-255) to use for channel colors.
            If 'auto' (the default), the original channel colors ` will be used, if available.
            If `None`, no channel colors will be assigned.

        interp_method : str
            Interpolation method used when warping slide. Default is "bicubic"

        tile_wh : int, optional
            Tile width and height used to save image

        compression : str, optional
            Compression method used to save ome.tiff . Default is lzw, but can also
            be jpeg or jp2k. See pyips for more details.

        Q : int
            Q factor for lossy compression

        """
        pathlib.Path(dst_dir).mkdir(exist_ok=True, parents=True)

        src_f_list = [self.original_img_list[slide_obj.stack_idx] for slide_obj in self.slide_dict.values()]

        cmap_is_str = False
        named_color_map = None
        if colormap is not None:
            if isinstance(colormap, str) and colormap == slide_io.CMAP_AUTO:
                cmap_is_str = True
            else:
                named_color_map = {self.get_slide(x).name:colormap[x] for x in colormap.keys()}

        for src_f in tqdm.tqdm(src_f_list, desc=SAVING_IMG_MSG, unit="image"):
            slide_obj = self.get_slide(src_f)
            slide_cmap = None
            is_rgb = slide_obj.reader.metadata.is_rgb
            if is_rgb:
                updated_channel_names = None
            elif colormap is not None:
                chnl_names = slide_obj.reader.metadata.channel_names
                updated_channel_names = slide_io.check_channel_names(chnl_names, is_rgb, nc=slide_obj.reader.metadata.n_channels)
                try:
                    if not cmap_is_str and named_color_map is not None:
                        slide_cmap = named_color_map[slide_obj.name]
                    else:
                        slide_cmap = colormap

                    slide_cmap = slide_io.check_colormap(colormap=slide_cmap, channel_names=updated_channel_names)
                except Exception as e:
                    traceback_msg = traceback.format_exc()
                    msg = f"Could not create colormap for the following reason:{e}"
                    valtils.print_warning(msg, traceback_msg=traceback_msg)

            dst_f = os.path.join(dst_dir, slide_obj.name + ".ome.tiff")

            slide_obj.warp_and_save_slide(dst_f=dst_f, level=level,
                                          non_rigid=non_rigid,
                                          crop=crop,
                                          src_f=slide_obj.src_f,
                                          interp_method=interp_method,
                                          colormap=slide_cmap,
                                          tile_wh=tile_wh,
                                          compression=compression,
                                          channel_names=updated_channel_names,
                                          Q=Q,
                                          pyramid=pyramid)


    @valtils.deprecated_args(perceputally_uniform_channel_colors="colormap")
    def warp_and_merge_slides(self, dst_f=None, level=0, non_rigid=True,
                              crop=True, channel_name_dict=None,
                              src_f_list=None, colormap=slide_io.CMAP_AUTO,
                              drop_duplicates=True, tile_wh=None,
                              interp_method="bicubic", compression="lzw",
                              Q=100, pyramid=True):

        """Warp and merge registered slides

        Parameters
        ----------
        dst_f : str, optional
            Path to were the warped slide will be saved. If None, then the slides will be merged
            but not saved.

        level : int, optional
            Pyramid level to be warped. Default is 0, which means the highest
            resolution image will be warped and saved.

        non_rigid : bool, optional
            Whether or not to conduct non-rigid warping. If False,
            then only a rigid transformation will be applied. Default is True

        crop: bool, str
            How to crop the registered images. If `True`, then the same crop used
            when initializing the `Valis` object will be used. If `False`, the
            image will not be cropped. If "overlap", the warped slide will be
            cropped to include only areas where all images overlapped.
            "reference" crops to the area that overlaps with the reference image,
            defined by `reference_img_f` when initialzing the `Valis object`.

        channel_name_dict : dict of lists, optional.
            key =  slide file name, value = list of channel names for that slide. If None,
            the the channel names found in each slide will be used.

        src_f_list : list of str, optionaal
            List of paths to slide to be warped. If None (the default), Valis.original_img_list
            will be used. Otherwise, the paths to which `src_f_list` points to should
            be an alternative copy of the slides, such as ones that have undergone
            processing (e.g. stain segmentation), had a mask applied, etc...

        colormap : list
            List of RGB colors (0-255) to use for channel colors

        drop_duplicates : bool, optional
            Whether or not to drop duplicate channels that might be found in multiple slides.
            For example, if DAPI is in multiple slides, then the only the DAPI channel in the
            first slide will be kept.

        tile_wh : int, optional
            Tile width and height used to save image

        interp_method : str
            Interpolation method used when warping slide. Default is "bicubic"

        compression : str
            Compression method used to save ome.tiff . Default is lzw, but can also
            be jpeg or jp2k. See pyips for more details.

        Q : int
            Q factor for lossy compression

        pyramid : bool
            Whether or not to save an image pyramid.

        Returns
        -------
        merged_slide : pyvips.Image
            Image with all channels merged. If `drop_duplicates` is True, then this
            will only contain unique channels.

        all_channel_names : list of str
            Name of each channel in the image

        ome_xml : str
            OME-XML string containing the slide's metadata

        """

        if channel_name_dict is not None:
            channel_name_dict_by_name = {valtils.get_name(k):channel_name_dict[k] for k in channel_name_dict}
        else:
            channel_name_dict_by_name = {slide_obj.name: [f"{c} ({slide_obj.name})" for c in slide_obj.reader.metadata.channel_names]
                                        for slide_obj in self.slide_dict.values()}

        if src_f_list is None:
            # Save in the sorted order. Will still be original order if imgs_ordered= True
            src_f_list = [self.original_img_list[slide_obj.stack_idx] for slide_obj in self.slide_dict.values()]

        all_channel_names = []
        merged_slide = None

        expected_channel_order = list(chain.from_iterable([channel_name_dict_by_name[valtils.get_name(f)] for f in src_f_list]))
        if drop_duplicates:
            expected_channel_order = list(dict.fromkeys(expected_channel_order))

        for f in src_f_list:
            slide_name = valtils.get_name(os.path.split(f)[1])
            slide_obj = self.slide_dict[slide_name]

            warped_slide = slide_obj.warp_slide(level, non_rigid=non_rigid,
                                                crop=crop,
                                                interp_method=interp_method)

            keep_idx = list(range(warped_slide.bands))
            slide_channel_names = channel_name_dict_by_name[slide_obj.name]

            if drop_duplicates:
                keep_idx = [idx for idx  in range(len(slide_channel_names)) if
                            slide_channel_names[idx] not in all_channel_names]

            if len(keep_idx) == 0:
                msg= f"Have already added all channels in {slide_channel_names}. Ignoring {slide_name}"
                valtils.print_warning(msg)
                continue

            if drop_duplicates and warped_slide.bands != len(keep_idx):
                keep_channels = [warped_slide[c] for c in keep_idx]
                slide_channel_names = [slide_channel_names[idx] for idx in keep_idx]
                if len(keep_channels) == 1:
                    warped_slide = keep_channels[0]
                else:
                    warped_slide = keep_channels[0].bandjoin(keep_channels[1:])
            print(f"merging {', '.join(slide_channel_names)} from {slide_obj.name}")

            if merged_slide is None:
                merged_slide = warped_slide
            else:
                merged_slide = merged_slide.bandjoin(warped_slide)

            all_channel_names.extend(slide_channel_names)

        if merged_slide.bands == 1:
            merged_slide = merged_slide.copy(interpretation="b-w")
        else:
            merged_slide = merged_slide.copy(interpretation="multiband")

        assert all_channel_names == expected_channel_order

        if colormap is not None:
            cmap_dict = slide_io.check_colormap(colormap, all_channel_names)
        else:
            cmap_dict = None

        slide_obj = self.get_ref_slide()
        px_phys_size = slide_obj.reader.scale_physical_size(level)
        bf_dtype = slide_io.vips2bf_dtype(merged_slide.format)
        out_xyczt = slide_io.get_shape_xyzct((merged_slide.width, merged_slide.height), merged_slide.bands)

        ome_xml_obj = slide_io.create_ome_xml(out_xyczt, bf_dtype, is_rgb=False,
                                              pixel_physical_size_xyu=px_phys_size,
                                              channel_names=all_channel_names,
                                              colormap=cmap_dict)
        ome_xml = ome_xml_obj.to_xml()

        if dst_f is not None:
            dst_dir = os.path.split(dst_f)[0]
            pathlib.Path(dst_dir).mkdir(exist_ok=True, parents=True)

            ref_slide = self.get_ref_slide()
            tile_wh = slide_io.get_tile_wh(reader=ref_slide.reader,
                                level=level,
                                out_shape_wh=out_xyczt[0:2])

            slide_io.save_ome_tiff(merged_slide, dst_f=dst_f,
                                   ome_xml=ome_xml,tile_wh=tile_wh,
                                   compression=compression, Q=Q, pyramid=pyramid)

        return merged_slide, all_channel_names, ome_xml



