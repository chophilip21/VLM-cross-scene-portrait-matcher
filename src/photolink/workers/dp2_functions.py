""""Add lower level dp2 functions to avoid cluttering the main functional modules."""

import os
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from loguru import logger

import photolink.utils.function as utils
from photolink import get_config
import photolink.models.fastreid as reid
import photolink.models.yolov11 as yolo
import photolink.models.scrfd as scrfd
import photolink.models.sface as sface
import photolink.models.florence as florence
from photolink.pipeline import get_cache_dir
from photolink.utils.function import search_all_images
from photolink.utils.image_loader import ImageLoader
import cv2
import IPython
import copy


# set glboal variables
config = get_config()
cache_dir = get_cache_dir() / Path(config["FASTREID"]["EMBEDDING_CACHE_DIR"])


def early_termination(error, pickle_cache):
    """Handle error and dump to pickle cache."""
    with open(pickle_cache, "wb") as f:
        data = {}
        data["error"] = error
        pickle.dump(data, f)


def debug_save_image(img, bounding_box, save_name):
    """Save image for debugging."""
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    bb = list(map(int, bounding_box))[:4]

    x1, y1, x2, y2 = bb
    # Draw a rectangle with a red color and thickness of 2
    cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

    cv2.imwrite(save_name, image)


def screen_bb_by_iou(yolo_preds: np.ndarray, florence_bboxes: list) -> np.ndarray:
    """
    Selects the Yolo prediction bounding box that has the highest IoU with the Florence-2 prediction.

    Parameters:
    -----------
    yolo_preds : np.ndarray
        An array of shape (N, 6) containing N Yolo predictions. Each prediction consists of:
        [x1, y1, x2, y2, confidence, class].
    florence_bboxes : list
    min_iou : float, optional
        The minimum IoU threshold required to accept a Yolo prediction.
        Defaults to 0.5.

    Returns:
    --------
    np.ndarray
        The Yolo prediction (1D array of length 6) with the highest IoU overlap with the Florence-2 prediction.

    Raises:
    -------
    ValueError
        If the Yolo predictions array has less than two predictions.
        If the Florence-2 prediction contains more than one bounding box.
        If the highest IoU is below the specified min_iou threshold.

    Notes:
    ------
    - The IoU (Intersection over Union) is computed between each Yolo bounding box and the single Florence-2 bounding box.
    - The function ensures that the Florence-2 prediction contains exactly one bounding box.
    """
    # Ensure Yolo predictions have more than one prediction
    if yolo_preds.shape[0] < 2:
        raise ValueError("Yolo predictions must contain more than one prediction.")

    if len(florence_bboxes) == 1:
        florence_bboxes = florence_bboxes[0]

    # Extract Florence bounding boxes
    if len(florence_bboxes) != 4:
        raise ValueError(
            "Florence prediction must contain exactly one bounding box of  x1, y1, x2, y2."
        )

    # Initialize variables to track the best IoU and corresponding Yolo prediction
    best_iou = 0
    best_yolo_pred = None

    # Compute IoU between Florence bbox and each Yolo bbox
    for yolo_pred in yolo_preds:
        yolo_bbox = yolo_pred[:4]  # [x1, y1, x2, y2]
        iou = compute_iou(yolo_bbox, florence_bboxes)
        # IPython.embed()

        if iou > best_iou:
            best_iou = iou
            best_yolo_pred = yolo_pred

    # Check if the best IoU meets the minimum threshold
    # if best_iou < min_iou:
    #     raise ValueError(f"No Yolo prediction meets the minimum IoU threshold of {min_iou}.")

    return best_yolo_pred


def compute_iou(box1, box2):
    """
    Computes the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    -----------
    box1 : list or np.ndarray
        Bounding box with format [x1, y1, x2, y2].
    box2 : list or np.ndarray
        Bounding box with format [x1, y1, x2, y2].

    Returns:
    --------
    float
        The IoU between the two bounding boxes.
    """
    # Convert to float for precision
    x1_min, y1_min, x1_max, y1_max = map(float, box1)
    x2_min, y2_min, x2_max, y2_max = map(float, box2)

    # Calculate the (x, y)-coordinates of the intersection rectangle
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)

    # Check if there is an overlap
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Compute the area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute the area of both bounding boxes
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    # Compute the IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou


def _precompute_embeddings(
    image_paths: List[str], debug: bool = False
) -> Tuple[List[Dict], set]:
    """Compute embeddings and return embeddings_info, and hash_set.

    Iterate over hash computed image paths, and store embeddings in cache_dir.
    """
    # Store all embeddings here
    cache_dir.mkdir(parents=True, exist_ok=True)
    embeddings_info = []
    hash_set = set()

    logger.info(f"Computing embeddings for {len(image_paths)} images.")

    for idx, img_path in enumerate(image_paths):
        # Use hash-based save and search to avoid overwriting similar image names
        path_hash = utils.path_to_hash(img_path)
        pickle_cache = cache_dir / f"{path_hash}.pkl"
        hash_set.add(pickle_cache)

        if not isinstance(img_path, str):
            raise ValueError("Image path must be a string. kill immediately.")

        # If the file already exists, load the embeddings
        if pickle_cache.exists():
            with open(pickle_cache, "rb") as f:
                data = pickle.load(f)
                embeddings_info.append(data)
        else:
            # Initialize data dictionary
            data = {}
            data["image_path"] = img_path
            data["hash"] = path_hash
            data["bbox"] = []
            data["reid_embedding"] = None

            try:
                image_loader = ImageLoader(img_path)
            except Exception as e:
                logger.error(f"Error loading image {img_path}, {idx}th image: {e}")
                early_termination(f"Error loading image {img_path}: {e}", pickle_cache)
                continue

            try:
                # Run person detection to get bounding boxes
                bounding_boxes = yolo.run_inference(image_loader)
            except Exception as e:
                logger.error(f"Error running yolo inference {img_path}: {e}")
                early_termination(
                    f"Error running yolo inference {img_path}: {e}", pickle_cache
                )
                continue

            # case 1. The length of bounding box is 0
            if len(bounding_boxes) == 0:
                logger.warning(
                    f"No bounding box detected in image {img_path}, skipping."
                )
                continue

            # case 2. The length of bounding box is exactly 1. Means this is the right person.
            elif len(bounding_boxes) == 1:
                best_prediction = bounding_boxes[0]

            else:
                # case 3. Most cases will fall here. You need to verify with Florence.
                florence_result = florence.run_inference(image_loader)
                florence_bboxes = florence_result.get("<OD>", {}).get("bboxes", [])

                # raise warning if florence result is more than one.
                if len(florence_bboxes) != 1:
                    logger.warning(
                        f"More than one bounding box detected in image {img_path}, Choosing first one."
                    )

                    florence_bboxes = florence_bboxes[0]

                best_prediction = screen_bb_by_iou(bounding_boxes, florence_bboxes)

            # Proceed with embeding calculation
            x1, y1, x2, y2, conf, cls_id = best_prediction
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Crop the image to the bounding box
            img = np.array(image_loader.get_downsampled_image())
            cropped_instance = copy.deepcopy(img)[y1:y2, x1:x2]

            # Get reid embedding
            try:
                reid_embedding = reid.get_reid_embedding(cropped_instance).squeeze()
            except Exception as e:
                logger.error(f"Error getting ReID embedding for {img_path}: {e}")
                early_termination(
                    f"Error getting ReID embedding for {img_path}: {e}",
                    pickle_cache,
                )
                continue

            # Append data
            data["bbox"] = best_prediction
            data["reid_embedding"] = reid_embedding

            # After processing all bounding boxes, store data
            embeddings_info.append(data)
            logger.info(f"Processed {idx+1}/{len(image_paths)} images.")

            # if debug, save image for debugging
            if debug:
                debug_path = os.path.join("test", os.path.basename(img_path))
                debug_save_image(img, florence_bboxes, debug_path)

        # Save data to cache
        with open(pickle_cache, "wb") as f:
            pickle.dump(data, f)

    return embeddings_info


def run(
    source_images: List[str], reference_images: List[str], debug: bool = False
) -> Tuple[Dict[int, np.ndarray], List[Dict]]:
    """Run the pipeline for the given images."""

    # if debug, remove cache dir everytime
    if debug:
        logger.info("Removing cache directory because debug is true.")
        shutil.rmtree(cache_dir, ignore_errors=True)
        shutil.rmtree("test", ignore_errors=True)
        os.makedirs("test", exist_ok=True)

    if not isinstance(source_images, list) or not isinstance(reference_images, list):
        raise ValueError("Input images must be a list of strings.")

    # Compute embeddings for source images
    source_embeddings_info = _precompute_embeddings(source_images, debug)


if __name__ == "__main__":
    print("Start to run the code")

    # on stage images
    # source_images = search_all_images(Path("~/for_phil/bcit_copy/a").expanduser())
    source_images = search_all_images(
        Path("/Users/philipcho/photomatcher/failure").expanduser()
    )

    # off stage images
    reference_images = search_all_images(Path("~/for_phil/bcit_copy/b").expanduser())

    # run pipeline
    run(source_images, reference_images, debug=True)
