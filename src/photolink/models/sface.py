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

    def align_crop_face(self, image, face) -> np.ndarray:
        """Crop the face from the image to fit size (112, 112, 3) using the face bounding box."""

        if not isinstance(image, np.ndarray):
            raise ValueError("image must be a numpy array.")

        if not isinstance(face, np.ndarray):
            raise ValueError("face must be a numpy array.")

        return self._model.alignCrop(image, face)

    def get_feat_from_aligned_face(self, aligned_face: np.ndarray):
        """Convert aligned face to feature vector, output is (1, 128)"""

        if not isinstance(aligned_face, np.ndarray):
            raise ValueError("aligned_face must be a numpy array.")

        if not aligned_face.shape == (112, 112, 3):
            raise ValueError("aligned_face must be a (112, 112, 3) numpy array.")

        return self._model.feature(aligned_face)

    def run_embedding_conversion(self, image: np.ndarray, faces: list) -> dict:
        """Run the embeddings conversion on all the faces, return list of embeddings per face."""

        result = {}
        embeddings_block = np.zeros((len(faces), 128))
        result["faces"] = []

        # run face recognition on all faces.
        for i, face in enumerate(faces):
            try:
                aligned_face = self.align_crop_face(image, face)
                result["faces"].append(aligned_face)
                feat_embedding = self.get_feat_from_aligned_face(aligned_face).squeeze()
                embeddings_block[i] = feat_embedding

            except Exception as e:
                result["error"] = e

        result["embeddings"] = embeddings_block

        return result


local = Local()


def get_embedding(image, faces):
    return local.model.run_embedding_conversion(image, faces)
