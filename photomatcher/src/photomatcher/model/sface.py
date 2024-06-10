"""Modules for Face recognition using Sface."""
import os

import cv2 as cv
import numpy as np


class Sface:
    """Face recognition model using Sface."""
    def __init__(self, modelPath, backendId=0, targetId=0):
        self._modelPath = modelPath
        self._backendId = backendId
        self._targetId = targetId

        self._model = cv.FaceRecognizerSF.create(
            model=self._modelPath, config="", backend_id=self._backendId, target_id=self._targetId
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

if os.getenv("SFACE_PATH") is None:
    raise ValueError("Please set the SFACENET_PATH in the environment variable.")


model_path = os.path.join(os.path.dirname(__file__), os.getenv("SFACE_PATH"))

if not os.path.exists(model_path):
    raise ValueError(f"Model path {model_path} does not exist.")

model = Sface(modelPath=model_path)

def get_sface():
    """Return the model."""
    return model

def choose_n_largest_face_yunet(faces: np.ndarray, faces_to_keep: int=3) -> np.ndarray:
    """Choose the top N largest face from the detected faces."""
    if faces.shape[0] == 0:
        raise ValueError("No faces detected.")

    if faces.shape[0] == 1:
        return faces

    # det[0:4] = face bounding box x,y,w,h
    if faces_to_keep > faces.shape[0]:
        faces_to_keep = faces.shape[0]

    faces = sorted(faces, key=lambda face: face[2] * face[3], reverse=True)[:faces_to_keep]

    return faces


def run_face_recognition(image: np.ndarray, faces: np.ndarray, faces_to_keep: int=3) -> dict:
    """Run the embeddings on the faces, return list of embeddings per face."""

    result = {}
    result["embeddings"] = []
    largest_faces = choose_n_largest_face_yunet(faces, faces_to_keep)

    try:
        for face in largest_faces:
            aligned_face = model.align_crop_face(image, face)
            feat_embedding = model.get_feat_from_aligned_face(aligned_face).squeeze()
            result["embeddings"].append(feat_embedding)

    except Exception as e:
        result["error"] = e

    return result
