"""Add lower level dp2 functions to avoid cluttering the main functional modules."""

import os
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from loguru import logger

import photolink.utils.function as utils
from photolink import get_config
from photolink.models.fastreid import get_reid_embedding, isolate_instance
from photolink.models.yolo_seg import get_segmentation
from photolink.pipeline import get_cache_dir
from photolink.utils.function import safe_load_image, search_all_images

# set glboal variables
config = get_config()
cache_dir = get_cache_dir() / Path(config["FASTREID"]["EMBEDDING_CACHE_DIR"])


def _precompute_embeddings(
    image_path: Union[str, List[str]]) -> Tuple[Dict[int, np.ndarray], List[Dict], set]:
    """Compute embeddings and return embeddings_info, and hash_set."""
    if isinstance(image_path, str):
        source_images = search_all_images(image_path)
    else:
        source_images = image_path

    # Store all embeddings here
    cache_dir.mkdir(parents=True, exist_ok=True)
    embeddings_info = []
    hash_set = set()

    for idx, img_path in enumerate(source_images):
        # Use hash-based save and search
        path_hash = utils.path_to_hash(img_path)
        pickle_cache = cache_dir / f"{path_hash}.pkl"
        hash_set.add(pickle_cache)

        if pickle_cache.exists():
            # Load embeddings from cache
            with open(pickle_cache, "rb") as f:
                embeddings = pickle.load(f)
                embeddings_info.extend(embeddings)
        else:
            try:
                img = safe_load_image(img_path)
                boxes, segments, masks = get_segmentation(img_path)
            except Exception as e:
                logger.error(f"Error processing image: {img_path}, {e}")
                continue

            tmp = []

            for i, ((*box, conf, cls_), mask) in enumerate(zip(boxes, masks)):
                # Crop instance and compute embedding
                try:
                    cropped_instance = isolate_instance(img, box, mask)
                    embedding = get_reid_embedding(cropped_instance)
                except Exception as e:
                    logger.error(f"Error processing instance: {e}")
                    continue

                embedding_info = {
                    "embedding": embedding,
                    "image_path": img_path,
                    "box_index": i,
                    "box": box,
                    "confidence": conf,
                    "class": cls_,
                    "mask": mask,
                }

                embeddings_info.append(embedding_info)
                tmp.append(embedding_info)

            # Save embeddings to cache
            with open(pickle_cache, "wb") as f:
                pickle.dump(tmp, f)

            logger.info(f"Saved embeddings to cache: {pickle_cache}")

    return embeddings_info, hash_set



def run_dp2_pipeline(source_path: str, target_path: str, debug: bool = True):
    """Run the entire DP2 pipeline."""
    # Run preprocess first
    source_images = search_all_images(source_path)
    target_images = search_all_images(target_path)
    all_images = source_images + target_images

    # Precompute embeddings just for source
    src_embeddings_info, src_hash_set = _precompute_embeddings(
        source_images, debug=debug
    )



if __name__ == "__main__":

    demo_path_a = str(Path(r"/Users/philipcho/for_phil/bcit_copy/a"))
    demo_path_b = str(Path(r"/Users/philipcho/for_phil/bcit_copy/b"))
    test = run_dp2_pipeline(demo_path_a, demo_path_b)
