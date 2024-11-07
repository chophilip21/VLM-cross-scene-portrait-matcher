import sys
from pathlib import Path
from typing import Union

import coremltools as ct
import cv2
import numpy as np
import onnxruntime as ort
from loguru import logger
from PIL import Image

from photolink import get_application_path, get_config
from photolink.models import Colors, class_names
from photolink.utils.function import check_weights_exist, safe_load_image


class Local:
    """Singleton class to manage the ONNX session and related configurations."""

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
        self.conf = None
        self.iou = None
        self.heuristic_threshold = None
        self.top_n = None
        self.class_names = None
        self.color_palette = None
        self._model = None

    @property
    def session(self):
        """Lazily initialize and return the ONNX session."""
        if self._session is None:
            application_path = get_application_path()
            config = get_config()

            model_path = str(
                application_path / Path(config.get("YOLOV11", "LOCAL_PATH"))
            )
            remote_path = str(config.get("YOLOV11", "REMOTE_PATH"))

            check_weights_exist(model_path, remote_path)

            # Start the session
            self._session = ort.InferenceSession(
                model_path,
                providers=(
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if ort.get_device() == "GPU"
                    else ["CPUExecutionProvider"]
                ),
            )
            self._input_name = self.session.get_inputs()[0].name
            self.set_metadata(config)

        return self._session

    @property
    def model(self):
        """Return CoreML model."""
        if self._model is None:
            application_path = get_application_path()
            config = get_config()
            model_path = str(
                application_path / Path(config.get("YOLOV11", "LOCAL_PATH_MAC"))
            )
            remote_path = str(config.get("YOLOV11", "REMOTE_PATH_MAC"))
            check_weights_exist(model_path, remote_path)
            self._model = ct.models.MLModel(model_path)
            self.set_metadata(config)

        return self._model

    def set_metadata(self, config):
        """Get metadata of the model."""
        self.width = int(config.get("YOLOV11", "WIDTH"))
        self.height = int(config.get("YOLOV11", "HEIGHT"))
        self.conf = float(config.get("YOLOV11", "CONF"))
        self.iou = float(config.get("YOLOV11", "IOU"))
        self.heuristic_threshold = float(config.get("YOLOV11", "HEURISTIC"))
        self.top_n = int(config.get("YOLOV11", "TOP_N"))
        self.class_names = class_names
        self.color_palette = Colors()


local = Local()  # Singleton instance of Local


def preprocess(img, model_height, model_width, ndtype):
    """Preprocess the input image for model inference."""
    # Check if the image is in RGB format (common for PIL images)
    if img.shape[2] == 3:
        # Convert from RGB to BGR if using OpenCV functions
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Resize image
    img_resized = cv2.resize(
        img, (model_width, model_height), interpolation=cv2.INTER_LINEAR
    )
    img_resized = img_resized.astype(ndtype) / 255.0

    # Transpose image to CHW format
    img_transposed = np.transpose(img_resized, (2, 0, 1))
    img_transposed = np.expand_dims(img_transposed, axis=0)  # Add batch dimension

    return img_transposed


def postprocess(
    output,
    img_shape,
    conf_threshold,
    iou_threshold,
    heuristic_threshold,
    num_candidates,
):
    """Post-process the model predictions to obtain bounding boxes."""
    # Extract outputs
    outputs = np.squeeze(output[0])
    outputs = np.transpose(outputs)

    # Get image dimensions
    img_height, img_width = img_shape[:2]

    # Lists to store the bounding boxes, scores, and class IDs of the detections
    boxes = []
    scores = []
    class_ids = []

    # Calculate the scaling factors for the bounding box coordinates
    x_factor = img_width / local.width
    y_factor = img_height / local.height

    # Iterate over each detection
    for i in range(outputs.shape[0]):
        # Extract the class scores from the current detection
        class_scores = outputs[i, 4:]
        max_score = np.max(class_scores)

        # If the maximum score is above the confidence threshold
        if max_score >= conf_threshold:
            # Get the class ID with the highest score
            class_id = np.argmax(class_scores)

            # Extract bounding box coordinates
            x, y, w, h = outputs[i, 0], outputs[i, 1], outputs[i, 2], outputs[i, 3]

            # Calculate the scaled coordinates of the bounding box
            left = int((x - w / 2) * x_factor)
            top = int((y - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)

            # Add the detection to the lists
            boxes.append([left, top, left + width, top + height])
            scores.append(max_score)
            class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)
    filtered_boxes = []
    filtered_scores = []
    filtered_class_ids = []

    indices = indices.flatten() if len(indices) > 0 else []

    for i in indices:
        filtered_boxes.append(boxes[i])
        filtered_scores.append(scores[i])
        filtered_class_ids.append(class_ids[i])

    # Convert to numpy arrays
    filtered_boxes = np.array(filtered_boxes)
    filtered_scores = np.array(filtered_scores)
    filtered_class_ids = np.array(filtered_class_ids)

    # Heuristic filtering
    if len(filtered_boxes) > 0:
        filtered_boxes = np.c_[filtered_boxes, filtered_scores, filtered_class_ids]
        filtered_boxes = heuristics_filter(
            filtered_boxes, img_shape, heuristic_threshold, num_candidates
        )

    return filtered_boxes


def heuristics_filter(boxes, im_shape, heuristic_threshold, num_candidates):
    """Filter boxes using heuristic scores based on size and distance from image center."""
    img_center = np.array([im_shape[1] / 2, im_shape[0] / 2])

    # Calculate the size of each bounding box
    box_sizes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # Calculate heuristic scores (normalized between 0 and 1)
    max_size = np.max(box_sizes) if len(box_sizes) > 0 else 1.0
    max_dist = np.linalg.norm(np.array([im_shape[1] / 2, im_shape[0] / 2]))

    scores = []
    for i in range(len(boxes)):
        box_center = np.array(
            [(boxes[i, 0] + boxes[i, 2]) / 2, (boxes[i, 1] + boxes[i, 3]) / 2]
        )
        size_penalty = (
            box_sizes[i] / max_size if max_size > 0 else 0
        )  # Avoid division by zero
        dist_from_center = np.linalg.norm(box_center - img_center)

        # Normalize distance penalty correctly
        dist_penalty = 1 - (
            dist_from_center / max_dist
        )  # Normalize distance penalty to [0,1]

        score = 0.5 * dist_penalty + 0.5 * size_penalty  # Weighted combination
        scores.append(score)

    scores = np.array(scores)

    # Sort boxes based on scores in descending order
    sorted_score_indices = np.argsort(-scores)
    boxes = boxes[sorted_score_indices]
    scores = scores[sorted_score_indices]

    # Apply dynamic penalty: only include top N if scores are relatively close
    filtered_boxes = []
    for i in range(len(boxes)):
        if i == 0:
            filtered_boxes.append(boxes[i])
        else:
            # Calculate relative score difference from the previous box
            prev_score = scores[i - 1]
            current_score = scores[i]

            # Only include if the current score is close enough to the previous score
            if abs(prev_score - current_score) < 0.15:  # Adjust the threshold as needed
                filtered_boxes.append(boxes[i])
            else:
                break  # Stop adding boxes if the score difference is too high

    # If there are more than num_candidates, select the top ones
    if len(filtered_boxes) > num_candidates:
        filtered_boxes = filtered_boxes[:num_candidates]

    return np.array(filtered_boxes)


def draw_and_visualize(im, boxes, save=True, name=None):
    """Draw bounding boxes and class labels on the image."""
    im_canvas = im.copy()
    for box in boxes:
        x1, y1, x2, y2, conf, cls_id = box
        cls_id = int(cls_id)
        color = local.color_palette(cls_id, bgr=True)

        # Draw bounding box rectangle
        cv2.rectangle(
            im_canvas,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            2,
            cv2.LINE_AA,
        )

        # Draw label
        label = f"{local.class_names[cls_id]}: {conf:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        label_x = int(x1)
        label_y = int(y1) - 10 if int(y1) - 10 > label_height else int(y1) + 10

        cv2.rectangle(
            im_canvas,
            (label_x, label_y - label_height),
            (label_x + label_width, label_y + label_height),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            im_canvas,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    # Convert image back to BGR if necessary
    if im_canvas.shape[2] == 3:
        im_canvas = cv2.cvtColor(im_canvas, cv2.COLOR_RGB2BGR)

    # Save image
    if save:
        if name is not None:
            cv2.imwrite(name, im_canvas)
        else:
            cv2.imwrite("test/demo.jpg", im_canvas)


def run_inference(
    input: Union[str, np.ndarray], debug=False, debug_path=None
) -> np.ndarray:
    """Perform inference on the input image."""

    if isinstance(input, str):
        original_image = safe_load_image(input)
    elif isinstance(input, np.ndarray):
        original_image = input
    else:
        raise ValueError(f"Input must be a string or numpy array, not {type(input)}")

    # Windows inference code
    if sys.platform == "win32":
        session = local.session  # Lazily initialize the session

        ndtype = (
            np.float16
            if session.get_inputs()[0].type == "tensor(float16)"
            else np.float32
        )

        # Preprocess image
        im = preprocess(original_image, local.height, local.width, ndtype)

        # Run inference
        preds = session.run(None, {session.get_inputs()[0].name: im})

    elif sys.platform == "darwin":
        # Mac inference code
        model = local.model
        im = preprocess(original_image, local.height, local.width, np.float32)

        im_input = np.squeeze(im).transpose(1, 2, 0)
        im_input = (im_input * 255).astype(np.uint8)
        im_input = Image.fromarray(im_input)
        core_ml_result = model.predict({"image": im_input})
        preds = []

        # Use the correct output key
        preds.append(core_ml_result["var_1230"])

    # Run post-processing
    boxes = postprocess(
        preds,
        original_image.shape,
        local.conf,
        local.iou,
        local.heuristic_threshold,
        local.top_n,
    )

    if debug:
        draw_and_visualize(
            original_image,
            boxes,
            save=True,
            name=debug_path,
        )

    return boxes


if __name__ == "__main__":
    import os
    from pathlib import Path

    from photolink.utils.function import search_all_images
    import IPython

    images = search_all_images(Path("~/for_phil/bcit_copy").expanduser())
    print(f"Found {len(images)} images.")

    for img in images:
        img_url = str(img)

        os.makedirs("test", exist_ok=True)
        debug_path = os.path.join("test", os.path.basename(img_url))
        print(f"Processing {img_url}...")

        # Inference
        boxes = run_inference(img_url, debug=True, debug_path=debug_path)
        logger.info(f"Detected {len(boxes)} instances in the image.")
        IPython.embed()
