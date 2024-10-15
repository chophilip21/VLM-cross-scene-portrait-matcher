from pathlib import Path
from typing import Union

import cv2
import numpy as np
import onnxruntime as ort
from loguru import logger

from photolink import get_application_path, get_config
from photolink.models import Colors, class_names
from photolink.utils.function import check_weights_exist, safe_load_image
import sys
import coremltools as ct
import IPython
from PIL import Image

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
        """Lazily initialize and return the ONNX session.

        Initializes the ONNX session and loads configuration parameters if not already initialized.

        Returns:
            onnxruntime.InferenceSession: The ONNX inference session.
        """
        if self._session is None:
            application_path = get_application_path()
            config = get_config()

            model_path = str(
                application_path / Path(config.get("YOLOSEG", "LOCAL_PATH"))
            )
            remote_path = str(config.get("YOLOSEG", "REMOTE_PATH"))

    
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
        """Return Coreml model."""

        if self._model is None:
            application_path = get_application_path()
            config = get_config()
            model_path = str(
                application_path / Path(config.get("YOLOSEG", "LOCAL_PATH_MAC"))
            )
            remote_path = str(config.get("YOLOSEG", "REMOTE_PATH_MAC"))
            check_weights_exist(model_path, remote_path)
            self._model = ct.models.MLModel(model_path)
            self.set_metadata(config)

        return self._model

    def set_metadata(self, config):
        """Get metadata of the model."""
        self.width = int(config.get("YOLOSEG", "WIDTH"))
        self.height = int(config.get("YOLOSEG", "HEIGHT"))
        self.conf = float(config.get("YOLOSEG", "CONF"))
        self.iou = float(config.get("YOLOSEG", "IOU"))
        self.heuristic_threshold = float(config.get("YOLOSEG", "HEURISTIC"))
        self.top_n = int(config.get("YOLOSEG", "TOP_N"))
        self.class_names = class_names
        self.color_palette = Colors()

local = Local()  # Singleton instance of Local


def preprocess(img, model_height, model_width, ndtype):
    """Preprocess the input image for model inference.

    Resizes and pads the image to fit the model's expected input size, normalizes pixel values,
    and rearranges dimensions as needed.

    Args:
        img (np.ndarray): The original image in HWC format.
        model_height (int): The height expected by the model.
        model_width (int): The width expected by the model.
        ndtype (np.dtype): The numpy data type for the model input (e.g., np.float32 or np.float16).

    Returns:
        Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
            - img_process (np.ndarray): The preprocessed image ready for inference.
            - ratio (Tuple[float, float]): The ratio used for resizing (width_ratio, height_ratio).
            - pad (Tuple[float, float]): The padding added to width and height (pad_w, pad_h).
    """
    # Resize and pad input image using letterbox
    shape = img.shape[:2]
    new_shape = (model_height, model_width)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    pad_w, pad_h = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    img = np.ascontiguousarray(np.einsum("HWC->CHW", img)[::-1], dtype=ndtype) / 255.0
    img_process = img[None] if len(img.shape) == 3 else img
    return img_process, ratio, (pad_w, pad_h)


def postprocess(
    preds,
    img,
    ratio,
    pad_w,
    pad_h,
    conf_threshold,
    iou_threshold,
    nm=32,
    human_only=True,
    num_candidates=-1,
):
    """Post-process the model predictions to obtain bounding boxes, segments, and masks.

    Args:
        preds (List[np.ndarray]): The raw outputs from the model (predictions and prototypes).
        img (np.ndarray): The original image.
        ratio (Tuple[float, float]): The resizing ratio (width_ratio, height_ratio).
        pad_w (float): The width padding applied during preprocessing.
        pad_h (float): The height padding applied during preprocessing.
        conf_threshold (float): Confidence threshold for filtering predictions.
        iou_threshold (float): IoU threshold for Non-Max Suppression (NMS).
        nm (int, optional): Number of masks. Defaults to 32.
        human_only (bool, optional): If True, only keep detections of humans (class index 0). Defaults to True.
        num_candidates (int, optional): Number of top candidates to keep after heuristic filtering. Defaults to -1.

    Returns:
        Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
            - boxes (np.ndarray): Array of detected bounding boxes and associated scores and class indices.
            - segments (List[np.ndarray]): List of segmentation contours for each detected object.
            - masks (np.ndarray): Array of binary masks for each detected object.
    """
    x, protos = preds[0], preds[1]

    if not isinstance(img, np.ndarray):
        img = np.array(img)

    # Transpose dimensions
    x = np.einsum("bcn->bnc", x)

    # Predictions filtering by confidence threshold
    x = x[np.amax(x[..., 4:-nm], axis=-1) > conf_threshold]

    # Combine boxes, scores, and classes into one matrix
    x = np.c_[
        x[..., :4],
        np.amax(x[..., 4:-nm], axis=-1),
        np.argmax(x[..., 4:-nm], axis=-1),
        x[..., -nm:],
    ]

    # NMS filtering
    x = x[cv2.dnn.NMSBoxes(x[:, :4], x[:, 4], conf_threshold, iou_threshold)]

    # Filter based on human_only flag
    if human_only:
        x = x[x[:, 5] == 0]  # Only keep class index 0 (human)

    if len(x) > 0:
        # Bounding boxes format change: cxcywh -> xyxy
        x[..., [0, 1]] -= x[..., [2, 3]] / 2
        x[..., [2, 3]] += x[..., [0, 1]]

        # Rescale boxes
        x[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
        x[..., :4] /= min(ratio)

        # Boundary clamp
        x[..., [0, 2]] = x[:, [0, 2]].clip(0, img.shape[1])
        x[..., [1, 3]] = x[:, [1, 3]].clip(0, img.shape[0])

        if human_only and num_candidates != -1:
            x = heuristics_filter(
                x, img.shape, local.heuristic_threshold, num_candidates
            )

        masks = process_mask(protos[0], x[:, 6:], x[:, :4], img.shape)
        segments = masks2segments(masks)

        return x[..., :6], segments, masks
    else:
        return [], [], []


def heuristics_filter(boxes, im_shape, heuristic_threshold, num_candidates):
    """Filter boxes using heuristic scores based on size and distance from image center.

    Calculates a heuristic score for each box, which is a weighted combination of its size and distance
    from the image center. Boxes with scores below the heuristic threshold are discarded unless no boxes
    meet the threshold, in which case the top candidate is returned.

    Args:
        boxes (np.ndarray): Array of bounding boxes with shape (N, 6+), where N is the number of boxes.
        im_shape (Tuple[int, int, int]): Shape of the image (height, width, channels).
        heuristic_threshold (float): Threshold for filtering based on heuristic scores.
        num_candidates (int): Maximum number of top candidates to keep.

    Returns:
        np.ndarray: Filtered array of bounding boxes.
    """
    img_center = np.array([im_shape[1] / 2, im_shape[0] / 2])

    # Calculate the size of each bounding box
    box_sizes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # Calculate heuristic scores (normalized between 0 and 1)
    max_size = np.max(box_sizes)
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

    # Filter based on heuristic threshold
    valid_indices = scores >= heuristic_threshold
    if np.any(valid_indices):
        # Select all valid candidates sorted by score
        filtered_boxes = boxes[valid_indices]
    else:
        # No candidates pass the threshold; return the top candidate
        filtered_boxes = boxes[:1]

    # If there are more than num_candidates, select the top ones
    if len(filtered_boxes) > num_candidates:
        filtered_boxes = filtered_boxes[:num_candidates]

    return filtered_boxes


def masks2segments(masks):
    """Convert binary masks to segmentation contours.

    Args:
        masks (np.ndarray): Array of binary masks with shape (N, H, W).

    Returns:
        List[np.ndarray]: List of segmentation contours for each mask.
    """
    segments = []
    for x in masks.astype("uint8"):
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        if c:
            c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
        else:
            c = np.zeros((0, 2))  # No segments found
        segments.append(c.astype("float32"))
    return segments


def crop_mask(masks, boxes):
    """Crop masks to their corresponding bounding boxes.

    Args:
        masks (np.ndarray): Array of masks with shape (N, H, W).
        boxes (np.ndarray): Array of bounding boxes with shape (N, 4).

    Returns:
        np.ndarray: Cropped masks, same shape as input masks.
    """
    n, h, w = masks.shape
    x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
    r = np.arange(w, dtype=x1.dtype)[None, None, :]
    c = np.arange(h, dtype=x1.dtype)[None, :, None]
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def process_mask(protos, masks_in, bboxes, img_shape):
    """Process and upscale masks to the original image size.

    Args:
        protos (np.ndarray): Prototype masks from the model output with shape (C, H, W).
        masks_in (np.ndarray): Mask coefficients with shape (N, C).
        bboxes (np.ndarray): Bounding boxes with shape (N, 4).
        img_shape (Tuple[int, int, int]): Shape of the original image (height, width, channels).

    Returns:
        np.ndarray: Array of binary masks with shape (N, img_height, img_width).
    """
    c, mh, mw = protos.shape
    masks = (
        np.matmul(masks_in, protos.reshape((c, -1)))
        .reshape((-1, mh, mw))
        .transpose(1, 2, 0)
    )  # HWN
    masks = np.ascontiguousarray(masks)
    masks = scale_mask(
        masks, img_shape
    )  # Re-scale mask from P3 shape to original input image shape
    masks = np.einsum("HWN -> NHW", masks)  # HWN -> NHW
    masks = crop_mask(masks, bboxes)
    return np.greater(masks, 0.5)


def scale_mask(masks, im0_shape, ratio_pad=None):
    """Rescale masks to the original image size.

    Args:
        masks (np.ndarray): Array of masks with shape (H1, W1, N) or (H1, W1).
        im0_shape (Tuple[int, int, int]): Shape of the original image (height, width, channels).
        ratio_pad (Tuple[float, float], optional): Ratio and padding used in preprocessing. Defaults to None.

    Returns:
        np.ndarray: Rescaled masks with shape matching the original image size.
    """
    im1_shape = masks.shape[:2]
    if ratio_pad is None:  # Calculate from im0_shape
        gain = min(
            im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1]
        )  # gain = old / new
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (
            im1_shape[0] - im0_shape[0] * gain
        ) / 2  # wh padding
    else:
        pad = ratio_pad[1]

    # Calculate top, left, bottom, right of mask
    top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))  # y, x
    bottom, right = int(round(im1_shape[0] - pad[1] + 0.1)), int(
        round(im1_shape[1] - pad[0] + 0.1)
    )
    if len(masks.shape) < 2:
        raise ValueError(
            f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}'
        )
    masks = masks[top:bottom, left:right]
    masks = cv2.resize(
        masks, (im0_shape[1], im0_shape[0]), interpolation=cv2.INTER_LINEAR
    )  # INTER_CUBIC would be better
    if len(masks.shape) == 2:
        masks = masks[:, :, None]
    return masks


def draw_and_visualize(im, bboxes, segments, save=True, name=None):
    """Draw bounding boxes, segmentation masks, and class labels on the image.

    Args:
        im (np.ndarray): The original image.
        bboxes (np.ndarray): Array of bounding boxes and associated scores and class indices.
        segments (List[np.ndarray]): List of segmentation contours for each detected object.
        save (bool, optional): Whether to save the resulting image. Defaults to True.
        name (str, optional): Filename to save the image. If None, defaults to 'test/demo.jpg'.

    Returns:
        None
    """
    
    # Draw rectangles and polygons
    im_canvas = im.copy()
    for (*box, conf, cls_), segment in zip(bboxes, segments):
        cls_ = int(cls_)

        # Draw contour and fill mask
        cv2.polylines(
            im, np.int32([segment]), True, (0, 255, 255), 2
        )  # Yellow borderline
        cv2.fillPoly(
            im_canvas, np.int32([segment]), local.color_palette(int(cls_), bgr=True)
        )

        # Draw bounding box rectangle
        cv2.rectangle(
            im,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            local.color_palette(int(cls_), bgr=True),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            im,
            f"{local.class_names[cls_]}: {conf:.3f}",
            (int(box[0]), int(box[1] - 9)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            local.color_palette(int(cls_), bgr=True),
            2,
            cv2.LINE_AA,
        )

    # Blend the original image with the mask overlay
    im = cv2.addWeighted(im_canvas, 0.3, im, 0.7, 0)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

    # Save image
    if save:
        if name is not None:
            cv2.imwrite(name, im)
        else:
            cv2.imwrite("test/demo.jpg", im)


def get_segmentation(
    input: Union[str, np.ndarray], debug=False, debug_path=None
) -> np.ndarray:
    """Perform segmentation on the input image.

    Args:
        input (Union[str, np.ndarray]): Input image or path to the image file.
        debug (bool, optional): If True, saves an image with visualizations. Defaults to False.
        debug_path (str, optional): Path to save the debug image if debug is True.

    Returns:
        List[Union[np.ndarray, List[np.ndarray]]]:
            - boxes (np.ndarray): Detected bounding boxes with scores and class indices.
            - segments (List[np.ndarray]): List of segmentation contours.
            - masks (np.ndarray): Array of binary masks.
    """

    original_image = safe_load_image(input)

    # windows inference code.
    if sys.platform == "win32":

        logger.info("Running segmentation inference on Windows platform.")
        session = local.session  # Lazily initialize the session

        session.ndtype = (
            np.half if session.get_inputs()[0].type == "tensor(float16)" else np.single
        )

        # Preprocess image
        im, ratio, (pad_w, pad_h) = preprocess(
            original_image, local.height, local.width, session.ndtype
        )

        # Run inference
        preds = session.run(None, {session.get_inputs()[0].name: im})

    elif sys.platform == "darwin":
        logger.info("Running segmentation inference on macOS platform.")
        # mac inference code.
        model = local.model
        im, ratio, (pad_w, pad_h) = preprocess(
            original_image, local.height, local.width, np.int8
        )

        im = np.squeeze(im).transpose(1, 2, 0) 
        im = (im * 255).astype(np.uint8)
        im = Image.fromarray(im)
        core_ml_result = model.predict({"image": im})
        preds = []

        # this is the correct order.
        preds.append(core_ml_result['var_1648'])
        preds.append(core_ml_result['p'])


    # Run post-processing
    boxes, segments, masks = postprocess(
        preds,
        original_image,
        ratio,
        pad_w,
        pad_h,
        local.conf,
        local.iou,
        nm=32,
        human_only=True,
        num_candidates=local.top_n,
    )

    if debug:
        draw_and_visualize(
            original_image,
            boxes,
            segments,
            save=True,
            name=debug_path,
        )

    return [boxes, segments, masks]


if __name__ == "__main__":
    
    # from ultralytics import YOLO
    # model = YOLO("yolo11m-seg.pt")
    # model.export(format="coreml", int8=True)
    # logger.info("Exported CoreML model")

    # Example usage
    import os
    import time
    from pathlib import Path

    # Define image path
    img_url = str(Path(r"sample/IMG_0066.JPG"))

    os.makedirs("test", exist_ok=True)
    debug_path = os.path.join("test", os.path.basename(img_url))

    # Inference
    box, segment, mask = get_segmentation(img_url, debug=True, debug_path=debug_path)
    logger.info(f"Detected {len(box)} instances in the image.")
