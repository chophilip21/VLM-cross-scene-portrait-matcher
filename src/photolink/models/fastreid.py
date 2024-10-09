"""FastReID model for extracting embeddings from images."""
import cv2
import numpy as np
import onnxruntime
from loguru import logger
from typing import Union
from photolink import get_application_path, get_config
from pathlib import Path

def preprocess(image_path: Union[str, np.ndarray], image_width: int, image_height: int):

    if isinstance(image_path, np.ndarray):
        original_image = image_path
    elif isinstance(image_path, str):
        original_image = cv2.imread(image_path)
    else:
        raise ValueError("image_path must be a string or numpy array")

    original_image = cv2.imread(image_path)
    # the model expects RGB inputs
    original_image = original_image[:, :, ::-1]

    # Apply pre-processing to image.
    img = cv2.resize(original_image, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
    img = img.astype("float32").transpose(2, 0, 1)[np.newaxis]  # (1, 3, h, w)
    return img

def normalize(nparray, order=2, axis=-1):
    """Normalize a N-D numpy array along the specified axis."""
    norm_val = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
    return nparray / (norm_val + np.finfo(np.float32).eps)

class Local:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Local, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self._session = None
        self.width = None
        self.height = None
        self._input_name = None

    @property
    def session(self):
        """Lazily initialize the ONNX session."""
        if self._session is None:
            application_path = get_application_path()
            config = get_config()
            model_path = str(application_path / Path(config.get("FASTREID", "LOCAL_PATH")))
            self.width = int(config.get("FASTREID", "WIDTH"))
            self.height = int(config.get("FASTREID", "HEIGHT"))

            # start the session
            self._session = onnxruntime.InferenceSession(model_path)
            self._input_name = self.session.get_inputs()[0].name

        return self._session

local = Local()  # Singleton instance of Local

def get_reid_embedding(input: Union[str, np.ndarray]) -> np.ndarray:
    """Get embeddings from an image using FastReID model."""
    session = local.session # Lazily initialize the session
    input_name = local._input_name
    image = preprocess(input, local.height, local.height)
    feat = session.run(None, {input_name: image})[0]
    feat = normalize(feat, axis=1)
    return feat

if __name__ == "__main__":
    pass
