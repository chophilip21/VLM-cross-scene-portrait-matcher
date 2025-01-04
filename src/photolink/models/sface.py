"""Modules for Face recognition using Sface."""

from pathlib import Path

import cv2 as cv
import numpy as np

from photolink import get_application_path, get_config
from photolink.utils.download import check_weights_exist


class Local:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Local, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self._model = None

    @property
    def model(self):
        if self._model is None:
            application_path = get_application_path()
            config = get_config()

            local_path = application_path / Path(config["SFACE"]["LOCAL_PATH"])
            remote_path = config["SFACE"]["REMOTE_PATH"]
            check_weights_exist(local_path, remote_path)

            self._model = Sface(modelPath=local_path)

        return self._model


class Sface:
    """Face recognition model using Sface."""

    def __init__(self, modelPath, backendId=0, targetId=0):
        self._modelPath = modelPath
        self._backendId = backendId
        self._targetId = targetId

        self._model = cv.FaceRecognizerSF.create(
            model=self._modelPath,
            config="",
            backend_id=self._backendId,
            target_id=self._targetId,
        )

    @property
    def name(self):
        return self.__class__.__name__

    def _align_crop_face(self, image, face) -> np.ndarray:
        """Crop the face from the image to fit size (112, 112, 3) using the face bounding box."""

        if not isinstance(image, np.ndarray):
            raise ValueError("image must be a numpy array.")

        if not isinstance(face, np.ndarray):
            raise ValueError("face must be a numpy array.")

        return self._model.alignCrop(image, face)

    def _get_feat_from_aligned_face(self, aligned_face: np.ndarray):
        """Convert aligned face to feature vector, output is (1, 128)"""

        if not isinstance(aligned_face, np.ndarray):
            raise ValueError("aligned_face must be a numpy array.")

        if not aligned_face.shape == (112, 112, 3):
            raise ValueError("aligned_face must be a (112, 112, 3) numpy array.")

        return self._model.feature(aligned_face)

    def _run_embedding_conversion(self, image: np.ndarray, faces: list) -> dict:
        """Run the embeddings conversion on all the faces, return list of embeddings per face.

        Image is required for aligning the faces properly."""

        result = {}
        embeddings_block = np.zeros((len(faces), 128))
        result["faces"] = []

        # run face recognition on all faces.
        for i, face in enumerate(faces):
            try:
                aligned_face = self._align_crop_face(image, face)
                result["faces"].append(aligned_face)
                feat_embedding = self._get_feat_from_aligned_face(
                    aligned_face
                ).squeeze()
                embeddings_block[i] = feat_embedding

            except Exception as e:
                result["error"] = e

        result["embeddings"] = embeddings_block

        return result


local = Local()


def get_sface_embedding(image: np.ndarray, faces: list) -> dict:
    """Generate SFace embeddings for detected faces in an image.

    This function uses the SFace model to compute feature embeddings for a list of faces
    detected in an image. Each face is aligned and cropped based on the provided bounding boxes
    before embedding extraction.

    Parameters:
    -----------
    image : np.ndarray
        The input image as a NumPy array. It is required for aligning the faces.
    faces : list
        A list of face bounding boxes, where each bounding box is represented as
        [x1, y1, x2, y2] in pixel coordinates.

    Returns:
    --------
    dict
        A dictionary containing:
        - 'faces': List of aligned face crops as NumPy arrays.
        - 'embeddings': NumPy array of shape (N, 128), where N is the number of faces,
          representing the embeddings for each face.
        - 'error': Exception details if an error occurs during processing (optional).

    Raises:
    -------
    ValueError
        If the input `image` is not a NumPy array or if `faces` is not a list of bounding boxes.
    """

    return local.model._run_embedding_conversion(image, faces)


if __name__ == "__main__":
    print("sface")
    from photolink.utils.image_loader import ImageLoader
    from photolink.models.scrfd import run_scrfd_inference
    import IPython

    im_loader = ImageLoader("sample/BCITCS24-C4P1-0008.JPG")
    bb = run_scrfd_inference(im_loader)

    # run sface
    test_result = get_sface_embedding(bb["image"], bb["faces"])

    IPython.embed()
