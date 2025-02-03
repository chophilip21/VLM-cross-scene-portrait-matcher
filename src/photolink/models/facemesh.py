import copy
from pathlib import Path

import cv2
import numpy as np
import onnxruntime
from loguru import logger
from PIL import Image
from typing import Union

from photolink import get_application_path, get_config
from photolink.models.scrfd import run_scrfd_inference
from photolink.utils.download import check_weights_exist
from photolink.utils.image_loader import ImageLoader

class Local:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Local, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self._session = None
        self._input_names = None
        self._output_names = None

    @property
    def session(self):
        """Lazy initialization of the FaceMesh onnx session."""
        if self._session is None:
            session_option_facemesh = onnxruntime.SessionOptions()
            session_option_facemesh.log_severity_level = 3

            # Read config for the model path
            application_path = get_application_path()
            config = get_config()
            model_path = application_path / Path(config["FACEMESH"]["LOCAL_PATH"])
            remote_path = str(config["FACEMESH"]["REMOTE_PATH"])

            downloaded = check_weights_exist(model_path, remote_path)
            if not downloaded:
                raise MemoryError("Could not download the FaceMesh model")

            logger.info(f"Loading FaceMesh model: {model_path}")
            self._session = onnxruntime.InferenceSession(
                model_path,
                sess_options=session_option_facemesh,
                providers=["CPUExecutionProvider"],
            )
            self._input_names = [inp.name for inp in self._session.get_inputs()]
            self._output_names = [out.name for out in self._session.get_outputs()]

        return self._session

    @property
    def input_names(self):
        if self._input_names is None:
            _ = self.session  # force session creation
        return self._input_names

    @property
    def output_names(self):
        if self._output_names is None:
            _ = self.session  # force session creation
        return self._output_names

# Instantiate the singleton
local = Local()

def extract_5_keypoints(landmarks_2d):
    """
    Convert 468 face mesh landmarks (each [x, y]) to 5 keypoints (2D only):
      1) Left eye center
      2) Right eye center
      3) Nose tip
      4) Left mouth corner
      5) Right mouth corner
    """
    LEFT_EYE_IDXS = [33, 133, 160, 158, 159, 144, 153, 145]
    RIGHT_EYE_IDXS = [263, 362, 387, 385, 386, 373, 380, 374]
    NOSE_TIP_IDX = 4
    LEFT_MOUTH_CORNER_IDX = 61
    RIGHT_MOUTH_CORNER_IDX = 291

    left_eye_points = landmarks_2d[LEFT_EYE_IDXS]
    right_eye_points = landmarks_2d[RIGHT_EYE_IDXS]
    nose_tip_point = landmarks_2d[NOSE_TIP_IDX]
    left_mouth_corner_point = landmarks_2d[LEFT_MOUTH_CORNER_IDX]
    right_mouth_corner_point = landmarks_2d[RIGHT_MOUTH_CORNER_IDX]

    left_eye_center = np.mean(left_eye_points, axis=0)  # [x, y]
    right_eye_center = np.mean(right_eye_points, axis=0)

    keypoints_2d = np.stack(
        [
            left_eye_center,
            right_eye_center,
            nose_tip_point,
            left_mouth_corner_point,
            right_mouth_corner_point,
        ],
        axis=0,
    )
    return keypoints_2d

def run_facemesh_inference(
    image: np.ndarray,
    face_box: Union[list, np.ndarray],  # [x1, y1, x2, y2]
    conf_threshold=0.95,
):
    """
    Runs FaceMesh inference on the given full BGR image and face bounding box.
    Returns a dictionary with the face detection score, full 2D landmarks (468,2),
    and the 5 2D keypoints if above conf_threshold.

    :param image: The full image in BGR format.
    :param face_box: [x1, y1, x2, y2] bounding box from face detector (SCRFD).
    :param conf_threshold: Confidence threshold for valid face mesh.
    :return: dict with:
        - "score": float
        - "landmarks_2d": np.ndarray of shape (468, 2)
        - "five_keypoints_2d": np.ndarray of shape (5, 2)
    """

    # If face_box is an array, flatten it to a list
    if isinstance(face_box, np.ndarray):
        face_box = face_box.squeeze()
        if not face_box.shape == (4,):
            raise ValueError(f"Invalid face_box shape. Expected (4,), not {face_box.shape}")
        face_box = face_box.tolist()

    x1, y1, x2, y2 = [int(v) for v in face_box]

    # Ensure bounding box is within image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)
    w = x2 - x1
    h = y2 - y1

    # If the image is a PIL image, convert to NumPy
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Prepare the 192x192 face input
    cropped_face = image[y1:y2, x1:x2, :]
    resized_face = cv2.resize(cropped_face, (192, 192))

    # Convert BGR->RGB and scale to [0..1]
    resized_face = resized_face[..., ::-1] / 255.0
    resized_face = resized_face.astype(np.float32)  # => (192,192,3)
    # NHWC -> NCHW
    resized_face = np.transpose(resized_face, (2, 0, 1))
    resized_face = np.expand_dims(resized_face, axis=0)  # (1,3,192,192)

    # Prepare bounding box arrays for the model
    np_crop_x1 = np.array([x1], dtype=np.int32).reshape(-1, 1)
    np_crop_y1 = np.array([y1], dtype=np.int32).reshape(-1, 1)
    np_crop_w = np.array([w], dtype=np.int32).reshape(-1, 1)
    np_crop_h = np.array([h], dtype=np.int32).reshape(-1, 1)

    # Run FaceMesh
    session = local.session
    input_names = local.input_names
    output_names = local.output_names

    # Typically: ['input_img', 'crop_x1', 'crop_y1', 'crop_width', 'crop_height']
    scores, final_landmarks = session.run(
        output_names,
        {
            input_names[0]: resized_face,
            input_names[1]: np_crop_x1,
            input_names[2]: np_crop_y1,
            input_names[3]: np_crop_w,
            input_names[4]: np_crop_h,
        },
    )

    # Build result dict
    result = {
        "score": None,
        "landmarks_2d": None,
        "five_keypoints_2d": None,
    }
    score = scores[0]
    result["score"] = score

    # Extract the 2D portion (dropping Z)
    lm_2d = final_landmarks[0][:, :2]
    result["landmarks_2d"] = lm_2d

    keypoints_2d = extract_5_keypoints(lm_2d)
    result["five_keypoints_2d"] = keypoints_2d
    

    # TODO: If the conf is too low, we can flag these later on. 

    # Print a warning if the confidence is below threshold
    if score < conf_threshold:
        logger.warning(f"FaceMesh confidence below threshold {score} < {conf_threshold})")

        # # sanity check
        # debug_image = image.copy()
        # # Draw the bounding box
        # cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # # Draw the 5 keypoints in RED
        # for (px, py) in keypoints_2d:
        #     cv2.circle(debug_image, (int(px), int(py)), 3, (0, 0, 255), -1)

        # # If you want to see the entire 468 landmarks as well
        # for (lx, ly) in lm_2d:
        #     cv2.circle(debug_image, (int(lx), int(ly)), 1, (255, 255, 0), -1)

        # debug_image = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)
        # cv2.imwrite("test_debug.jpg", debug_image)
        # import IPython
        # IPython.embed()

    return result


if __name__ == "__main__":
    im_loader = ImageLoader("sample/IMG_0066.JPG")
    ds_image = np.array(im_loader.get_downsampled_image())  # BGR

    # Detect face with SCRFD
    detection_result = run_scrfd_inference(ds_image, heuristic_filter=True)
    face_bounding_box = detection_result["faces"].tolist()[0]  # [x1, y1, x2, y2]
    logger.info(f"Detected face bounding box: {face_bounding_box}")

    # Run facemesh inference
    mesh_result = run_facemesh_inference(
        ds_image, face_bounding_box, conf_threshold=0.95
    )
    print(mesh_result)
    score = mesh_result["score"]
    logger.info(f"FaceMesh Score: {score}")
