
"""Modules for Face detection using yunet."""
import numpy as np
import cv2
import os

class YuNet:
    """Face detection model using Yunet."""
    def __init__(self, modelPath, inputSize=[320, 320], confThreshold=0.6, nmsThreshold=0.3, topK=5000, backendId=0, targetId=0):
        self._modelPath = modelPath
        self._inputSize = tuple(inputSize) # [w, h]
        self._confThreshold = confThreshold
        self._nmsThreshold = nmsThreshold
        self._topK = topK
        self._backendId = backendId
        self._targetId = targetId

        self._model = cv2.FaceDetectorYN.create(
            model=self._modelPath,
            config="",
            input_size=self._inputSize,
            score_threshold=self._confThreshold,
            nms_threshold=self._nmsThreshold,
            top_k=self._topK,
            backend_id=self._backendId,
            target_id=self._targetId)

    @property
    def name(self):
        return self.__class__.__name__

    def setInputSize(self, input_size):
        self._model.setInputSize(tuple(input_size))

    def infer(self, image)-> np.ndarray:
        """Inference output is the np.array of bounding box and landmarks."""
        faces = self._model.detect(image)
        return np.array([]) if faces[1] is None else faces[1]
    

# TODO: CHECK WITH MULTIPLE FACES. WE NEED TO DEAL WITH THIS PART.
backend_target_pairs = [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU]
backend_id = backend_target_pairs[0]
target_id = backend_target_pairs[1]
conf_threshold = float(os.getenv("YUNET_CONF", 0.75))
nms_threshold = float(os.getenv("YUNET_NMS", 0.4))
top_k = 5000 ## bb to keep before nms

if os.getenv("YUNET_PATH") is None:
    raise ValueError("Please set the YUNET_PATH in the environment variable.")

model_path = os.path.join(os.path.dirname(__file__), os.getenv("YUNET_PATH"))

if not os.path.exists(model_path):
    raise ValueError(f"Model path {model_path} does not exist.")

# singleton entry point.
input_size = int(os.getenv("YUNET_INPUT_SIZE", 640))
model = YuNet(modelPath=model_path,
                    inputSize=[input_size, input_size],
                    confThreshold=conf_threshold,
                    nmsThreshold=nms_threshold,
                    topK=top_k,
                    backendId=backend_id,
                    targetId=target_id)


def get_yunet():
    """Return the model."""
    return model


def run_face_detection(image_path: str) -> dict:
    """Run face detection on the image using Yunet."""

    face_table = {}

    try:
        image = cv2.imread(image_path)
    except Exception as e:
        print(f"Error reading image {image_path}. Error: {e}")
        face_table['error'] = f"Error reading image {image_path}. Error: {e}"
        return face_table


    if image.shape[0] > 1000:
        image = cv2.resize(image, (0, 0),
                           fx=500 / image.shape[0], fy=500 / image.shape[0])

    h, w, _ = image.shape
    face_table['image'] = image # if you need the resized image.

    # Inference results saved to dict.
    try: 
        model.setInputSize([w, h])
        results = model.infer(image)

        # no face detected. Just return with out face key.
        if results.size == 0:
            return face_table

        face_table['faces'] = results
    except Exception as e:
        print(f"Error running inference on image {image_path}. Error: {e}")
        face_table['error'] = f"Error running inference on image {image_path}. Error: {e}"

    return face_table
