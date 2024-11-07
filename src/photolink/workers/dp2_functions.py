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
from photolink.pipeline import get_cache_dir
from photolink.utils.function import safe_load_image, search_all_images
import IPython
import cv2

# set glboal variables
config = get_config()
cache_dir = get_cache_dir() / Path(config["FASTREID"]["EMBEDDING_CACHE_DIR"])


def early_termination(error, pickle_cache):
    """Handle error and dump to pickle cache."""
    with open(pickle_cache, "wb") as f:
        data = {}
        data["error"] = error
        pickle.dump(data, f)


def debug_save_image(img, path):
    """Save image for debugging."""
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    path_ = os.path.join("debug", os.path.basename(path))
    cv2.imwrite(path_, image)


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
            data["reid_embedding"] = []
            data["face_embedding"] = []

            try:
                img = safe_load_image(img_path)
            except Exception as e:
                logger.error(f"Error loading image {img_path}: {e}")
                early_termination(f"Error loading image {img_path}: {e}", pickle_cache)
                continue

            try:
                # Run person detection to get bounding boxes
                bounding_boxes = yolo.run_inference(img)
            except Exception as e:
                logger.error(f"Error running yolo inference {img_path}: {e}")
                early_termination(
                    f"Error running yolo inference {img_path}: {e}", pickle_cache
                )
                continue

            for box in bounding_boxes:
                x1, y1, x2, y2, conf, cls_id = box
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                # Crop the image to the bounding box
                cropped_instance = img[y1:y2, x1:x2]

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

                # Run face detection on the cropped instance
                face_dets = scrfd.run_inference(cropped_instance)
                if "error" in face_dets:
                    e = face_dets["error"]
                    logger.error(f"Error running SCRFD on {img_path}: {e}")
                    early_termination(
                        f"Error running SCRFD on {img_path}: {e}", pickle_cache
                    )
                    continue

                if "faces" not in face_dets:
                    # No face detected in this bounding box, skip
                    logger.warning(
                        f"No face detected in bounding box {box} in image {img_path}, skipping."
                    )

                    if debug:
                        debug_save_image(cropped_instance, img_path)
                    continue

                face_dets = face_dets["faces"]

                # Get face embeddings
                try:
                    face_embeddings_result = sface.get_embedding(
                        cropped_instance, face_dets[:, :4]
                    )
                    if "embeddings" not in face_embeddings_result:
                        logger.warning(
                            f"No face embeddings obtained for bounding box {box} in image {img_path}, skipping."
                        )
                        continue

                    face_embeddings = face_embeddings_result["embeddings"]

                except Exception as e:
                    logger.error(f"Error getting face embedding for {img_path}: {e}")
                    early_termination(
                        f"Error getting face embedding for {img_path}: {e}",
                        pickle_cache,
                    )
                    continue

                # Append data
                data["bbox"].append(box)
                data["reid_embedding"].append(reid_embedding)
                data["face_embedding"].append(face_embeddings)

            # After processing all bounding boxes, store data
            embeddings_info.append(data)

            # Save data to cache
            with open(pickle_cache, "wb") as f:
                pickle.dump(data, f)

    return embeddings_info, hash_set


def run(
    source_images: List[str], reference_images: List[str], debug: bool = False
) -> Tuple[Dict[int, np.ndarray], List[Dict]]:
    """Run the pipeline for the given images."""

    # if debug, remove cache dir everytime
    if debug:
        logger.info("Removing cache directory because debug is true.")
        shutil.rmtree(cache_dir, ignore_errors=True)

    if not isinstance(source_images, list) or not isinstance(reference_images, list):
        raise ValueError("Input images must be a list of strings.")

    # Compute embeddings for source images
    source_embeddings_info, source_hash_set = _precompute_embeddings(
        source_images, debug
    )
    logger.info(f"Source embeddings computed for {len(source_images)} images.")

    # Compute embeddings for reference images
    reference_embeddings_info, reference_hash_set = _precompute_embeddings(
        reference_images, debug
    )
    logger.info(f"Reference embeddings computed for {len(reference_images)} images.")

    IPython.embed()


if __name__ == "__main__":
    print("Start to run the code")

    # on stage images
    source_images = search_all_images(Path("~/for_phil/bcit_copy/a").expanduser())

    # off stage images
    reference_images = search_all_images(Path("~/for_phil/bcit_copy/b").expanduser())

    # run pipeline
    run(source_images, reference_images, debug=True)
