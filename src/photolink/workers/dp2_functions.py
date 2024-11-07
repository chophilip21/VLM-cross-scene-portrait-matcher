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
from photolink.models.fastreid import get_reid_embedding, isolate_instance
from photolink.models.yolov11 import run_inference
from photolink.pipeline import get_cache_dir
from photolink.utils.function import safe_load_image, search_all_images

# set glboal variables
config = get_config()
cache_dir = get_cache_dir() / Path(config["FASTREID"]["EMBEDDING_CACHE_DIR"])


def _precompute_embeddings(
    image_path: Union[str, List[str]]
) -> Tuple[Dict[int, np.ndarray], List[Dict], set]:
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


if __name__ == "__main__":
    print("Start to run the code")
