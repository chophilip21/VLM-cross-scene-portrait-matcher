"""Modules for Face recognition using Sface."""

from pathlib import Path

import cv2 as cv
import numpy as np

from photolink import get_application_path, get_config
from photolink.utils.download import check_weights_exist
import math


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


def _x_dist(face_box, center_x):
    x1, y1, x2, y2 = face_box
    box_center_x = (x1 + x2) / 2.0
    return abs(box_center_x - center_x)

def get_sface_embedding(image: np.ndarray, faces: np.ndarray, heuristic_filter=False) -> dict:
    """Generate SFace embeddings for detected faces in an image.

    Parameters:
    -----------
    image : np.ndarray
        The input image as a NumPy array. It is required for aligning the faces.
    faces : np.ndarray
        An array of face bounding boxes, where each bounding box is represented as
        [x1, y1, x2, y2] in pixel coordinates.

    heuristic_filter : bool, optional
        Whether to only return embeddings for the single "best" face.

    Returns:
    --------
    dict
        A dictionary containing:
        - 'faces': List of aligned face crops as NumPy arrays.
        - 'embeddings': NumPy array of shape (N, 128), where N is the number of faces,
          representing the embeddings for each face.
        - 'error': Exception details if an error occurs during processing (optional).
    """
    
    # decide if heuristic filter should be applied when multiple faces are detected
    if len(faces) > 1 and heuristic_filter:

        # what matters is the width (x-axis)
        _, img_width = image.shape[:2]
        center_x = img_width / 2.0

        sorted_by_xdist = sorted(faces, key=lambda box: _x_dist(box, center_x))
        closest_box = sorted_by_xdist[0]
        closest_area = (closest_box[2] - closest_box[0]) * (closest_box[3] - closest_box[1])

        """"
        We care about box in the middle of x-axis the most.
        But just in case, if the closest box is too small, we will check if there's any box that is significantly bigger and could be a better candidate.
        """
        bigger_candidates = []
        for box in sorted_by_xdist[1:]:
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            if area >= 1.5 * closest_area:
                bigger_candidates.append((box, area))

        if not bigger_candidates:
            # Nothing is significantly bigger, use closest_box
            faces = np.array([closest_box], dtype=np.float32)
        else:
            # Pick the single largest among bigger_candidates
            best_box, best_area = max(bigger_candidates, key=lambda item: item[1])
            faces = np.array([best_box], dtype=np.float32)

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
