"""Efficient image loader."""

import copy
from io import BytesIO
from typing import Union
from PIL import Image, ImageOps
from photolink import get_config
from pathlib import Path
import copy

class ImageLoader:
    def __init__(self, image_path: str):
        """
        Initialize the ImageLoader with the path to the image and the desired downsample size.
        """
        self.image_path = image_path
        self._original_image = self.safe_load_image(self.image_path)
        self._downsampled_image = None
        self._scale_x = 1.0
        self._scale_y = 1.0

        # should be based on config size of yolo model. 
        config = get_config()
        ds_width = int(config.get("YOLOV11", "WIDTH"))
        ds_height = int(config.get("YOLOV11", "HEIGHT"))
        self.downsample_size = (ds_width, ds_height)


    def _copy_image_meta(self, src: Image.Image, dest: Image.Image):
        """
        Copy metadata from the source image to the destination image.
        """
        preserve_metadata_keys = ["info", "icc_profile", "exif", "dpi", "applist", "format"]
        for key in preserve_metadata_keys:
            if hasattr(src, key):
                setattr(dest, key, copy.deepcopy(getattr(src, key)))
        return dest

    def safe_load_image(self, image: Union[bytes, str]) -> Image.Image:
        """
        Load an image from bytes or a file path, ensuring the orientation is correct.
        """
        # Ensure image is bytes or a valid file path
        if isinstance(image, str):
            
            if not Path(image).exists():
                raise FileNotFoundError(f"Image file not found: {image}")

            with open(image, "rb") as f:
                image = f.read()
        elif not isinstance(image, bytes):
            raise TypeError(f"Image must be bytes or a file path, not {type(image)}")

        pil_image = Image.open(BytesIO(image))

        # Correct orientation if necessary
        if hasattr(pil_image, "_getexif") and pil_image._getexif() is not None:
            new_pil_image = ImageOps.exif_transpose(pil_image)
            pil_image = self._copy_image_meta(pil_image, new_pil_image)

        return pil_image

    def get_original_image(self) -> Image.Image:
        """
        Get a copy of the original image.
        """
        return self._original_image.copy()

    def get_downsampled_image(self) -> Image.Image:
        """
        Downsample the image to the specified size while maintaining aspect ratio.
        """
        if self._downsampled_image is not None:
            return self._downsampled_image

        original_width, original_height = self._original_image.size

        # Determine target size
        if isinstance(self.downsample_size, int):
            target_width = target_height = self.downsample_size
        else:
            target_width, target_height = self.downsample_size

        # Compute scaling ratio to maintain aspect ratio
        ratio = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)

        # Calculate scaling factors for coordinate mapping
        self._scale_x = original_width / new_width
        self._scale_y = original_height / new_height

        # Downsample the image
        downsampled_image = self._original_image.resize((new_width, new_height), Image.LANCZOS)
        downsampled_image = self._copy_image_meta(self._original_image, downsampled_image)

        self._downsampled_image = downsampled_image

        return self._downsampled_image

    @property
    def scale_x(self) -> float:
        """
        Accessor for the scale factor in the x-dimension from downsampled to original image.
        """
        return self._scale_x

    @property
    def scale_y(self) -> float:
        """
        Accessor for the scale factor in the y-dimension from downsampled to original image.
        """
        return self._scale_y
