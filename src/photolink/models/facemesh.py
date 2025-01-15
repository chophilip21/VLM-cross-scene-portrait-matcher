import onnxruntime
import numpy as np
import cv2
from photolink.models.scrfd import run_scrfd_inference
from photolink.utils.image_loader import ImageLoader
from loguru import logger
import copy
from photolink import get_application_path, get_config
from pathlib import Path
from photolink.utils.download import check_weights_exist


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
            check_weights_exist(model_path, remote_path)

            logger.info(f"Loading FaceMesh model: {model_path}")
            self._session = onnxruntime.InferenceSession(
                model_path,
                sess_options=session_option_facemesh,
                providers=[
                    "CPUExecutionProvider",
                ],
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


def extract_5_keypoints(landmarks_3d):
    """
    Convert 468 face mesh landmarks to 5 keypoints:
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

    left_eye_points = landmarks_3d[LEFT_EYE_IDXS]
    right_eye_points = landmarks_3d[RIGHT_EYE_IDXS]
    nose_tip_point = landmarks_3d[NOSE_TIP_IDX]
    left_mouth_corner_point = landmarks_3d[LEFT_MOUTH_CORNER_IDX]
    right_mouth_corner_point = landmarks_3d[RIGHT_MOUTH_CORNER_IDX]

    left_eye_center = np.mean(left_eye_points, axis=0)
    right_eye_center = np.mean(right_eye_points, axis=0)

    keypoints_3d = np.stack(
        [
            left_eye_center,
            right_eye_center,
            nose_tip_point,
            left_mouth_corner_point,
            right_mouth_corner_point,
        ],
        axis=0,
    )

    return keypoints_3d


def visualize_landmarks(image, landmarks, color=(0, 255, 0), radius=1, thickness=-1):
    """
    Draw 2D circles for each (x, y, z) in `landmarks` onto `image`.
    """
    image = copy.deepcopy(image)
    for lx, ly, lz in landmarks:
        cv2.circle(image, (int(lx), int(ly)), radius, color, thickness)
    return image


def run_facemesh_inference(
    image: np.ndarray, face_box: list, conf_threshold=0.95  # [x1, y1, x2, y2]
):
    """
    Runs FaceMesh inference on the given (cropped) face region.
    Returns a dictionary with the face detection score, 468 landmarks,
    and the 5 keypoints (if above conf_threshold), etc.

    :param image: The full image in BGR format.
    :param face_box: [x1, y1, x2, y2] bounding box from face detector (SCRFD).
    :param conf_threshold: Confidence threshold for valid face mesh.
    :return: dict with:
        - "score": float
        - "landmarks_3d": np.ndarray of shape (468, 3) if score >= threshold, else None
        - "five_keypoints_3d": np.ndarray of shape (5, 3) if score >= threshold, else None
    """
    x1, y1, x2, y2 = [int(v) for v in face_box]
    w = x2 - x1
    h = y2 - y1

    # Crop & Resize to 192x192
    cropped_face = image[y1:y2, x1:x2, :]
    resized_face = cv2.resize(cropped_face, (192, 192))
    # BGR->RGB and scale to [0..1]
    resized_face = resized_face[..., ::-1] / 255.0
    # (192,192,3)->float32->(1,3,192,192)
    resized_face = resized_face.astype(np.float32)
    resized_face = np.transpose(resized_face, (2, 0, 1))
    resized_face = np.expand_dims(resized_face, axis=0)

    # Prepare bounding box arrays for the model
    np_crop_x1 = np.array([x1], dtype=np.int32).reshape(-1, 1)
    np_crop_y1 = np.array([y1], dtype=np.int32).reshape(-1, 1)
    np_crop_w = np.array([w], dtype=np.int32).reshape(-1, 1)
    np_crop_h = np.array([h], dtype=np.int32).reshape(-1, 1)

    # Run FaceMesh
    session = local.session
    input_names = local.input_names
    output_names = local.output_names

    # Typically: input_names -> [ 'input_img', 'crop_x1', 'crop_y1', 'crop_width', 'crop_height' ]
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

    result = {
        "score": None,
        "landmarks_3d": None,
        "five_keypoints_3d": None,
    }
    score = scores[0]
    result["score"] = score

    if score >= conf_threshold:
        # final_landmarks shape => (1, 468, 3)
        lm_3d = final_landmarks[0]
        result["landmarks_3d"] = lm_3d

        # Convert 468 => 5
        keypoints_3d = extract_5_keypoints(lm_3d)
        result["five_keypoints_3d"] = keypoints_3d
    else:
        logger.warning(f"FaceMesh confidence below threshold: {score:.4f}")

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

    if mesh_result["landmarks_3d"] is not None:
        # Visualize the 468-landmarks or the 5 keypoints
        figure = visualize_landmarks(
            ds_image, mesh_result["landmarks_3d"], color=(0, 255, 0)
        )
        cv2.imwrite("output_468.jpg", figure)

        # Also show the 5 keypoints in red
        kpts_5 = mesh_result["five_keypoints_3d"]  # shape (5,3)
        for px, py, pz in kpts_5:
            cv2.circle(figure, (int(px), int(py)), 3, (0, 0, 255), -1)

        cv2.imwrite("output_5.jpg", figure)
    else:
        logger.warning(
            "FaceMesh inference was too low-confidence to produce landmarks."
        )
