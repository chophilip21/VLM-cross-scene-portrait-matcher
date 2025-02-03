import copy
import os
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import IPython
import nmslib
import numpy as np
from loguru import logger

import photolink.models.florence as florence
import photolink.models.scrfd as scrfd
from photolink.models.facemesh import run_facemesh_inference
from photolink.models.facetransformer import run_face_recognition
import photolink.models.yolov11 as yolo
import photolink.utils.function as utils
from photolink import get_config
from photolink.pipeline import get_cache_dir
from photolink.utils.function import search_all_images
from photolink.utils.image_loader import ImageLoader


# set glboal variables
config = get_config()
cache_dir = get_cache_dir() / Path(config["FASTREID"]["EMBEDDING_CACHE_DIR"])


def _early_termination(error, pickle_cache):
    """Handle error and dump to pickle cache."""
    with open(pickle_cache, "wb") as f:
        data = {}
        data["error"] = error
        pickle.dump(data, f)


def _debug_save_image(img, bounding_box, save_name):
    """Save image for debugging."""
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if bounding_box is None:
        cv2.imwrite(save_name, image)
        return
    if isinstance(bounding_box, np.ndarray):
        bounding_box = bounding_box.tolist()

    # the output of coreml and onnx is a bit different
    if len(bounding_box) == 6:
        bb = [int(x) for x in bounding_box[:4]]  # is flat list
    else:
        bb = [int(x) for x in bounding_box[0]]  # is nested list

    x1, y1, x2, y2 = bb
    # Draw a rectangle with a red color and thickness of 2

    cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

    cv2.imwrite(save_name, image)


def _screen_bb_by_iou(yolo_preds: np.ndarray, florence_bboxes: list) -> np.ndarray:
    """
    Selects the Yolo prediction bounding box that has the highest IoU with the Florence-2 prediction.

    Parameters:
    -----------
    yolo_preds : np.ndarray
        An array of shape (N, 6) containing N Yolo predictions. Each prediction consists of:
        [x1, y1, x2, y2, confidence, class].
    florence_bboxes : list

    Returns:
    --------
    np.ndarray
        The Yolo prediction (1D array of length 6) with the highest IoU overlap with the Florence-2 prediction.

    Raises:
    -------
    ValueError
        If the Yolo predictions array has less than two predictions.
        If the Florence-2 prediction contains more than one bounding box.
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

        if iou > best_iou:
            best_iou = iou
            best_yolo_pred = yolo_pred

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


def _precompute_embeddings(image_paths: List[str], debug: bool = False) -> Dict:
    """
    Computes and caches embeddings for a list of image paths.

    Parameters:
    -----------
    image_paths : List[str]
        List of image file paths to process.
    debug : bool, optional
        If True, saves debug images with bounding boxes. Defaults to False.

    Returns:
    --------
    Dict : Dict mapping image hashes to embedding information.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    embeddings_info = {}

    for idx, img_path in enumerate(image_paths):
        # Use hash-based save and search to avoid overwriting similar image names
        path_hash = utils.path_to_hash(img_path)
        pickle_cache = cache_dir / f"{path_hash}.pkl"

        if not isinstance(img_path, str):
            raise ValueError("Image path must be a string. kill immediately.")

        # If the file already exists, load the embeddings
        if pickle_cache.exists():
            with open(pickle_cache, "rb") as f:
                data = pickle.load(f)
                embeddings_info[path_hash] = data

        else:
            # Initialize data dictionary
            data = {}
            data["image_path"] = img_path
            data["bbox"] = []
            data["face_embedding"] = None

            try:
                image_loader = ImageLoader(img_path)
            except Exception as e:
                logger.error(f"Error loading image {img_path}, {idx}th image: {e}")
                _early_termination(f"Error loading image {img_path}: {e}", pickle_cache)
                continue

            try:
                # Run person detection to get bounding boxes
                bounding_boxes = yolo.run_inference(image_loader)
            except Exception as e:
                logger.error(f"Error running yolo inference {img_path}: {e}")
                _early_termination(
                    f"Error running yolo inference {img_path}: {e}", pickle_cache
                )
                continue

            # case 1. The length of bounding box is 0. We need to skip this case.
            if len(bounding_boxes) == 0:
                logger.warning(
                    f"No bounding box detected in image {img_path}, skipping."
                )
                continue

            # case 2. The length of bounding box is exactly 1, do not run florence.
            elif len(bounding_boxes) == 1:
                best_prediction = bounding_boxes[0]

            else:
                # case 3. Multiple bounding boxes detected: run Florence and confirm with IOU
                florence_result = florence.run_inference(image_loader)
                florence_bboxes = florence_result.get("<OD>", {}).get("bboxes", [])

                # If multiple bboxes from Florence, choose the first one (no change in logic)
                if len(florence_bboxes) != 1:
                    raise ValueError(
                        f"Florence prediction must contain exactly one bounding box. It detected {len(florence_bboxes)} bounding boxes."
                    )

                best_prediction = _screen_bb_by_iou(bounding_boxes, florence_bboxes)

                if best_prediction is None:
                    raise ValueError(
                        "No matching bounding box found from YOLO for Florence bounding box."
                    )

            # Proceed with embedding calculation
            x1, y1, x2, y2, conf, cls_id = best_prediction
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Crop the image to the bounding box
            img = np.array(image_loader.get_downsampled_image())
            cropped_instance = copy.deepcopy(img)[y1:y2, x1:x2]

            # run face detection. Limit to 1 face.
            face_table = scrfd.run_scrfd_inference(
                cropped_instance, heuristic_filter=True
            )

            if "faces" not in face_table or len(face_table["faces"]) == 0:
                logger.warning(f"No face detected in {img_path}, skipping.")
                continue

            # run face mesh calculation
            face_mesh = run_facemesh_inference(cropped_instance, face_table["faces"])

            # get face embedding
            face_embedding = run_face_recognition(
                cropped_instance, face_mesh["five_keypoints_2d"]
            )

            # Append data (embedding, keypoint, etc)
            data["bbox"] = best_prediction
            data["face_embedding"] = face_embedding
            data["landmarks_2d"] = face_mesh["landmarks_2d"]
            data["kpts_5"] = face_mesh["five_keypoints_2d"]

            # After processing all bounding boxes, store data
            embeddings_info[path_hash] = data
            logger.info(f"Processed {idx+1}/{len(image_paths)} images.")

            # for debug mode, confirm things are correct.
            if debug:
                debug_path = os.path.join("test", os.path.basename(img_path))
                _debug_save_image(cropped_instance, face_table["faces"], debug_path)

        # Save data to cache for each image.
        with open(pickle_cache, "wb") as f:
            pickle.dump(data, f)

    return embeddings_info


def run_dp2_pipeline(
    source_path: str, reference_path: str, debug: bool = False
) -> Tuple[Dict[int, np.ndarray], List[Dict]]:
    """Run the pipeline for the given images."""

    # if debug, remove cache dir everytime
    if debug:
        logger.info("Removing cache directory because debug is true.")
        shutil.rmtree(cache_dir, ignore_errors=True)
        shutil.rmtree("test", ignore_errors=True)
        os.makedirs("test", exist_ok=True)

    source_images = search_all_images(Path(source_path).expanduser())
    reference_images = search_all_images(Path(reference_path).expanduser())

    # Compute embeddings for source images
    source_embeddings_info = _precompute_embeddings(source_images, debug)
    # Compute embeddings for reference images
    reference_embeddings_info = _precompute_embeddings(reference_images, debug)

    # 1) Initialize the nmslib index
    index = nmslib.init(method="hnsw", space="l2")
    src_vectors = []
    idx_to_source_key = []

    for src_key, src_data in source_embeddings_info.items():
        src_embed = src_data["face_embedding"]
        src_vectors.append(src_embed)
        idx_to_source_key.append(src_key)

    index.addDataPointBatch(src_vectors)
    index.createIndex({"post": 2}, print_progress=False)
    index.setQueryTimeParams({"efSearch": 100})

    # Keep track of which source indices are already matched (1:1 usage)
    used_indices = set()
    # Keep track of references that couldn't match a source
    unmatched_refs = {}

    results = {}

    # 2) Perform the matching, iterate over reference embeddings
    for ref_key, ref_data in reference_embeddings_info.items():
        ref_embed = ref_data["face_embedding"]

        # Skip if no embedding
        if ref_embed is None:
            logger.warning(f"No embedding for reference {ref_key}. Skipping.")
            unmatched_refs[ref_key] = ref_data["image_path"]
            continue

        # Query for all possible neighbors, sorted by ascending distance
        nbrs_idx, nbrs_dist = index.knnQuery(ref_embed, k=len(src_vectors))

        best_idx = None
        best_dist = None

        # Find the first neighbor not in used_indices
        for i, candidate_idx in enumerate(nbrs_idx):
            if candidate_idx not in used_indices:
                best_idx = candidate_idx
                best_dist = float(nbrs_dist[i])
                break

        if best_idx is None:
            # All sources used or not suitable
            logger.info(f"No available source left for reference {ref_key}.")
            unmatched_refs[ref_key] = ref_data["image_path"]
            continue

        # Mark this source as used
        used_indices.add(best_idx)

        # Map nmslib index back to source key
        best_src_key = idx_to_source_key[best_idx]
        ref_path = ref_data["image_path"]
        src_path = source_embeddings_info[best_src_key]["image_path"]

        # Store the match
        results[best_src_key] = {
            "source_path": src_path,
            "reference_path": ref_path,
            "distance": best_dist,
        }

    # Collect sources that were NEVER matched by any reference
    unused_sources = {}
    for i, src_key in enumerate(idx_to_source_key):
        if i not in used_indices:
            unused_sources[src_key] = source_embeddings_info[src_key]["image_path"]

    logger.info("Finished matching references to sources.")
    logger.info(f"Found {len(results)} matched pairs.")
    logger.info(f"{len(unused_sources)} source images got no match.")
    logger.info(f"{len(unmatched_refs)} references had no matching source.")

    IPython.embed()

    # Return or handle however you'd like
    return results, unused_sources, unmatched_refs


if __name__ == "__main__":
    print("Start to run the code")

    # Windows
    # source_images = r"C:\Users\choph\photomatcher\dataset\subset\stage"
    # reference_images = r"C:\Users\choph\photomatcher\dataset\subset\off"

    # Mac
    source_images = r"~/for_phil/bcit_copy/a"
    reference_images = r"~/for_phil/bcit_copy/b"

    run_dp2_pipeline(source_images, reference_images, debug=False)
