"""Modules for Face detection using SCRFD."""

from pathlib import Path
from typing import Union

import cv2
import numpy as np
from loguru import logger
from onnxruntime import InferenceSession

from photolink import get_application_path, get_config
from photolink.models.exceptions import NoFaceDetectedError
from photolink.utils.download import check_weights_exist
from photolink.utils.image_loader import ImageLoader


class Local:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Local, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self._model = None
        self.confidence_threshold = None
        self.nms_threshold = None

    @property
    def model(self):
        if self._model is None:
            application_path = get_application_path()
            config = get_config()
            self.confidence_threshold = float(config["SCRFD"]["SCRFD_CONF_THRESHOLD"])
            self.nms_threshold = float(config["SCRFD"]["SCRFD_NMS_THRESHOLD"])
            model_path = application_path / Path(config["SCRFD"]["LOCAL_PATH"])
            remote_path = str(config["SCRFD"]["REMOTE_PATH"])
            check_weights_exist(model_path, remote_path)

            self._model = SCRFD(str(model_path))

        return self._model


class SCRFD:
    """Face detection model using SCRFD."""

    def __init__(self, model_path: str, providers=None):
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.session = InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[
            0
        ].shape  # Get expected input shape
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self.num_anchors = 2

    @staticmethod
    def _distance2bbox(points, distance):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        return np.stack([x1, y1, x2, y2], axis=-1)

    @staticmethod
    def _nms(dets, thresh):
        x1, y1, x2, y2, scores = (
            dets[:, 0],
            dets[:, 1],
            dets[:, 2],
            dets[:, 3],
            dets[:, 4],
        )
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep

    def infer(
        self, img: np.ndarray, threshold: float, nms_threshold: float
    ) -> np.ndarray:
        input_blob = self._preprocess(img)
        outputs = self.session.run(self.output_names, {self.input_name: input_blob})

        scores_list = []
        bboxes_list = []

        input_height, input_width = img.shape[:2]

        for idx, stride in enumerate(self._feat_stride_fpn):
            # Extract the outputs for scores and bbox predictions
            # Adjust the indices based on the model's outputs
            scores = outputs[idx]
            bbox_preds = outputs[idx + self.fmc] * stride

            # Reshape scores and bbox_preds appropriately
            # Assuming the outputs have shape [batch_size, num_anchors, height, width]
            # We need to reshape them to [num_anchors * height * width]
            scores = scores.squeeze(0)  # Remove batch dimension if present
            bbox_preds = bbox_preds.squeeze(0)

            # Reshape scores and bbox_preds to 1D arrays
            scores = scores.reshape(-1)
            bbox_preds = bbox_preds.reshape(-1, 4)

            height = input_height // stride
            width = input_width // stride

            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(
                np.float32
            )
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            if self.num_anchors > 1:
                anchor_centers = np.repeat(anchor_centers, self.num_anchors, axis=0)

            pos_inds = np.where(scores >= threshold)[0]
            if len(pos_inds) == 0:
                continue
            pos_scores = scores[pos_inds]
            pos_bbox_preds = bbox_preds[pos_inds]
            pos_anchor_centers = anchor_centers[pos_inds]

            decoded_bboxes = self._distance2bbox(pos_anchor_centers, pos_bbox_preds)
            scores_list.append(pos_scores)
            bboxes_list.append(decoded_bboxes)

        if len(scores_list) == 0:
            return np.array([])

        scores = np.concatenate(scores_list, axis=0)
        bboxes = np.concatenate(bboxes_list, axis=0)
        dets = np.hstack((bboxes, scores[:, np.newaxis]))
        keep = self._nms(dets, nms_threshold)
        dets = dets[keep]
        return dets

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)
        img -= 127.5
        img *= 1.0 / 128.0
        img = img.transpose(2, 0, 1)  # Change data layout from HWC to CHW
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img

    def run_face_detection(
        self,
        image_loader: Union[ImageLoader, np.ndarray],
        conf_threshold: float,
        nms_threshold: float,
    ) -> dict:
        """Perform face detection on an image using the SCRFD model.

        Parameters:
        -----------
        image_loader : Union[ImageLoader, np.ndarray]
            The image to process, either as an `ImageLoader` instance or a NumPy array.
        conf_threshold : float
            Confidence threshold for filtering detections.
        nms_threshold : float
            Non-Maximum Suppression (NMS) threshold for filtering overlapping bounding boxes.

        Returns:
        --------
        dict
            A dictionary containing:
            - 'resize_ratio': The scaling factor applied during preprocessing.
            - 'image': The resized and padded image used for inference (optional).
            - 'faces': Array of detected face bounding boxes [x1, y1, x2, y2] (optional).
            - 'error': Description of any errors encountered (optional).

        Raises:
        -------
        ValueError
            If the input image_loader is neither an ImageLoader instance nor a NumPy array.

        NoFaceDetectedError
            If no faces are detected in the image.
        """

        face_table = {"resize_ratio": 1.0}

        if isinstance(image_loader, np.ndarray):
            image = image_loader
        elif isinstance(image_loader, ImageLoader):
            image = np.array(image_loader.get_downsampled_image())

        else:
            raise ValueError(
                "Image loader must be an instance of ImageLoader or np.ndarray."
            )

        # Preprocess the image according to SCRFD requirements

        img_height, img_width = image.shape[:2]
        input_height, input_width = self.input_shape[2], self.input_shape[3]

        # Calculate scale factor while keeping aspect ratio
        scale = min(input_width / img_width, input_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        face_table["resize_ratio"] = 1 / scale  # To scale back coordinates later

        # Resize the image
        resized_img = cv2.resize(image, (new_width, new_height))

        # Create a new image and paste the resized image into it
        det_img = np.zeros((input_height, input_width, 3), dtype=np.uint8)
        det_img[0:new_height, 0:new_width, :] = resized_img

        face_table["image"] = det_img  # Include the resized and padded image if needed

        dets = self.infer(det_img, conf_threshold, nms_threshold)

        # No face detected. Just return without 'faces' key.
        if dets.size == 0:
            raise NoFaceDetectedError("No faces detected.")

        # Adjust bounding boxes back to original image scale
        dets[:, :4] /= scale
        # Extract bounding boxes [x1, y1, x2, y2]
        face_table["faces"] = dets[:, :4]

        return face_table


local = Local()


def _x_dist(face_box, center_x):
    x1, _, x2, _ = face_box
    box_center_x = (x1 + x2) / 2.0
    return abs(box_center_x - center_x)


def run_scrfd_inference(
    image_loader: Union[ImageLoader, np.ndarray], heuristic_filter: bool = False
) -> dict:
    """Run face detection on the image using SCRFD.

    Faces is a list where each face is represented as [x1, y1, x2, y2]. Apply heuristic filter to get 1 face.
    """
    result_dict = local.model.run_face_detection(
        image_loader, local.confidence_threshold, local.nms_threshold
    )

    if heuristic_filter:

        faces = result_dict.get("faces")

        if faces.ndim == 2 and faces.shape[0] > 1:

            # what matters is the width (x-axis)
            _, img_width = image_loader.shape[:2]
            center_x = img_width / 2.0

            sorted_by_xdist = sorted(faces, key=lambda box: _x_dist(box, center_x))
            closest_box = sorted_by_xdist[0]

            closest_area = (closest_box[2] - closest_box[0]) * (
                closest_box[3] - closest_box[1]
            )

            """"
            We care about box in the middle of x-axis the most.
            But just in case, if the closest box is too small, we will check if there's any box that is significantly bigger and could be a better candidate.
            """
            bigger_candidates = []

            for box in sorted_by_xdist[1:]:
                x1, y1, x2, y2 = box
                area = (x2 - x1) * (y2 - y1)
                if area >= 1.25 * closest_area:
                    bigger_candidates.append((box, area))

            if not bigger_candidates:
                # Nothing is significantly bigger, use closest_box
                faces = np.array([closest_box], dtype=np.float32)
            else:
                # Pick the single largest among bigger_candidates
                best_box, best_area = max(bigger_candidates, key=lambda item: item[1])
                faces = np.array([best_box], dtype=np.float32)
                logger.info("favoring bigger face over closest face")

            # overwrite the keys for faces
            result_dict["faces"] = faces

    return result_dict


if __name__ == "__main__":
    print("Starting SCRFD face detection...")

    im_loader = ImageLoader("sample/IMG_0066.JPG")
    test_result = run_scrfd_inference(im_loader)
    print(test_result)
