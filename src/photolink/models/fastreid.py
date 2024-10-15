"""FastReID model for extracting embeddings from images."""

from pathlib import Path
from typing import Union

import cv2
import numpy as np
import onnxruntime

from photolink import get_application_path, get_config
from photolink.utils.function import check_weights_exist


class Local:
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

    @property
    def session(self):
        """Lazily initialize the ONNX session."""
        if self._session is None:
            application_path = get_application_path()
            config = get_config()
            model_path = str(
                application_path / Path(config.get("FASTREID", "LOCAL_PATH"))
            )
            remote_path = str(config.get("FASTREID", "REMOTE_PATH"))

            # Check if the model weights exist
            check_weights_exist(model_path, remote_path)

            self.width = int(config.get("FASTREID", "WIDTH"))
            self.height = int(config.get("FASTREID", "HEIGHT"))

            # start the session
            self._session = onnxruntime.InferenceSession(model_path)
            self._input_name = self.session.get_inputs()[0].name

        return self._session


local = Local()  # Singleton instance of Local


def preprocess(input: Union[str, np.ndarray], image_width: int, image_height: int):

    if isinstance(input, np.ndarray):
        original_image = input
    elif isinstance(input, str):
        original_image = cv2.imread(input)
    else:
        raise ValueError("image_path must be a string or numpy array")

    # the model expects RGB inputs
    original_image = original_image[:, :, ::-1]

    # Apply pre-processing to image.
    img = cv2.resize(
        original_image, (image_width, image_height), interpolation=cv2.INTER_CUBIC
    )
    img = img.astype("float32").transpose(2, 0, 1)[np.newaxis]  # (1, 3, h, w)
    return img


def normalize(nparray, order=2, axis=-1):
    """Normalize a N-D numpy array along the specified axis."""
    norm_val = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
    return nparray / (norm_val + np.finfo(np.float32).eps)


def get_reid_embedding(input: Union[str, np.ndarray]) -> np.ndarray:
    """Get embeddings from an image using FastReID model."""
    session = local.session  # Lazily initialize the session
    input_name = local._input_name
    image = preprocess(input, local.width, local.height)

    feat = session.run(None, {input_name: image})[0]
    feat = normalize(feat, axis=1)
    return feat


def isolate_instance(
    image: np.ndarray, box: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """
    Isolate the instance in the image using the bounding box and mask.

    Args:
        image (np.ndarray): The input image.
        box (np.ndarray): Bounding box [x1, y1, x2, y2] for cropping.
        mask (np.ndarray): Binary mask of the instance with the same dimensions as the image.

    Returns:
        np.ndarray: The cropped image with background outside the mask set to zero.
    """
    # Extract bounding box coordinates and ensure they are integers
    x1, y1, x2, y2 = map(int, box[:4])

    # Ensure coordinates are within image boundaries
    h, w = image.shape[:2]
    x1 = np.clip(x1, 0, w)
    x2 = np.clip(x2, 0, w)
    y1 = np.clip(y1, 0, h)
    y2 = np.clip(y2, 0, h)

    # Crop the image and mask using the bounding box
    cropped_image = image[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]

    # Ensure the mask is binary (0 or 255)
    binary_mask = (cropped_mask > 0).astype(np.uint8) * 255

    # Apply the mask to the cropped image
    isolated_instance = cv2.bitwise_and(cropped_image, cropped_image, mask=binary_mask)

    return isolated_instance


if __name__ == "__main__":
    import os
    from collections import defaultdict

    import hdbscan
    import pandas as pd
    from loguru import logger
    from sklearn.cluster import AgglomerativeClustering

    from photolink.models.yolo_seg import get_segmentation
    from photolink.utils.function import safe_load_image, search_all_images

    conf = 0.5
    iou = 0.45

    source_images = search_all_images("/Users/philipcho/for_phil/bcit_copy")

    embeddings_info = []

    for idx, img_path in enumerate(source_images):
        assert os.path.exists(img_path), f"Image path not found: {img_path}"

        try:
            img = safe_load_image(img_path)
            boxes, segments, masks = get_segmentation(img_path)
        except Exception as e:
            logger.error(f"Error processing image: {img_path}, {e}")
            continue

        logger.info(f"mode detected {len(boxes)} instances on path: {img_path}")

        for i, ((*box, conf, cls_), mask) in enumerate(zip(boxes, masks)):

            # crop instance, convert it into embedding
            try:
                cropped_instance = isolate_instance(img, box, mask)
            except Exception as e:
                # in actual code, deal with this better.
                logger.error(f"Error cropping instance: {e}")
                continue

            embedding = get_reid_embedding(cropped_instance)

            embeddings_info.append(
                {
                    "embedding": embedding,
                    "image_path": img_path,
                    "box_index": i,
                    "box": box,
                    "confidence": conf,
                    "class": cls_,
                    "mask": mask,
                }
            )

    logger.info("Embeddings extracted successfully.")

    # Convert embeddings to numpy array for clustering
    embeddings = np.array([item["embedding"] for item in embeddings_info])
    embeddings = np.squeeze(embeddings, axis=1)

    # Decide which clustering algorithm to use based on data size
    if len(embeddings) < 20:
        # Use AgglomerativeClustering for small datasets
        from sklearn.cluster import AgglomerativeClustering

        cluster_obj = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5)
        labels = cluster_obj.fit_predict(embeddings)
    else:
        # Use HDBSCAN for larger datasets
        cluster_obj = hdbscan.HDBSCAN(min_cluster_size=2, leaf_size=50)
        labels = cluster_obj.fit_predict(embeddings)

    # Associate cluster labels back to embeddings_info
    for idx, label in enumerate(labels):
        embeddings_info[idx]["cluster_label"] = label

    # Convert to Pandas DataFrame
    # embeddings_df = pd.DataFrame(embeddings_info)

    # Create a directory to save annotated images
    output_dir = "clustered_images"
    os.makedirs(output_dir, exist_ok=True)

    # Group embeddings_info by image_path
    image_entries = defaultdict(list)
    for entry in embeddings_info:
        image_entries[entry["image_path"]].append(entry)

    # Now for each image, draw bounding boxes and cluster labels
    for image_path, entries in image_entries.items():
        # Load the image
        img = safe_load_image(image_path)

        # For each entry, draw bounding box and cluster label
        for entry in entries:
            box = entry["box"]
            cluster_label = entry["cluster_label"]
            x1, y1, x2, y2 = map(int, box[:4])

            # Draw the bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

            # Prepare the text for the cluster label
            text = f"Cluster {cluster_label}"
            # Set font and get text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.9
            thickness = 2
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_width, text_height = text_size

            # Set the text background rectangle coordinates
            text_offset_x = x1
            text_offset_y = y1 - 10
            box_coords = (
                (text_offset_x, text_offset_y - text_height - 5),
                (text_offset_x + text_width + 5, text_offset_y + 5),
            )

            # Draw rectangle behind text
            cv2.rectangle(img, box_coords[0], box_coords[1], (0, 255, 0), cv2.FILLED)
            # Put the cluster label text above the bounding box
            cv2.putText(
                img,
                text,
                (text_offset_x, text_offset_y),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
            )

        # Save the annotated image
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, img)
        logger.info(f"Annotated image saved to {output_path}")
