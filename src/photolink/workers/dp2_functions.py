"""Add lower level dp2 functions to avoid cluttering the main functional modules."""

import os
from pathlib import Path

from loguru import logger

from photolink import get_config
from photolink.utils.function import safe_load_image, search_all_images
from photolink.pipeline import get_cache_dir
from photolink.models.yolo_seg import get_segmentation
from photolink.models.fastreid import isolate_instance, get_reid_embedding
import pickle
import numpy as np
import hdbscan
import photolink.utils.function as utils
from collections import Counter
import IPython
from typing import Union
from photolink.workers.draw import embeddings_sanity_check


def _preprocess_embeddings(image_path: Union[str, list]) -> list:
    """Use yoloworld to detect humans based on precomputed embeddings.

    Iterate over list of images, and return dict of duplicate centroids.
    """

    if isinstance(image_path, str):
        source_images = search_all_images(image_path)
    else:
        source_images = image_path

    # store all he embeddings here.
    config = get_config()
    cache_dir = get_cache_dir() / Path(config.get("FASTREID", "EMBEDDING_CACHE_DIR"))
    cache_dir.mkdir(exist_ok=True)

    # start iterating over the images.
    embeddings_info = []
    hash_set = set()

    for idx, img_path in enumerate(source_images):

        # use hash based save and search.
        path_hash = utils.path_to_hash(img_path)
        pickle_cache = cache_dir / Path(f"{path_hash}.pkl")
        hash_set.add(pickle_cache)

        # if the pickle cache exists, avoid inference by loading.
        if pickle_cache.exists():

            # this will be a list
            with open(pickle_cache, "rb") as f:
                embeddings = pickle.load(f)

                if idx % 200 == 0:
                    logger.info(f"Loading embeddings from cache...")

                for embedding in embeddings:
                    embeddings_info.append(embedding)
        else:
            try:
                img = safe_load_image(img_path)

                # TODO, Maybe make this return heuristic scores.
                boxes, segments, masks = get_segmentation(img_path)
            except Exception as e:
                logger.error(f"Error processing image: {img_path}, {e}")
                continue

            tmp = []

            for i, ((*box, conf, cls_), mask) in enumerate(zip(boxes, masks)):

                # crop instance, convert it into embedding
                try:
                    cropped_instance = isolate_instance(img, box, mask)
                except Exception as e:
                    # in actual code, deal with this better.
                    logger.error(f"Error cropping instance: {e}")
                    continue

                embedding = get_reid_embedding(cropped_instance)

                # first append data to embeddings_info
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

                # append the same data to tmp just for saving purpose.
                tmp.append(
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

            # dump the embeddings to the cache.
            with open(pickle_cache, "wb") as f:
                pickle.dump(tmp, f)

            logger.info(f"Saved embeddings to cache: {pickle_cache}")

    # run clustering on the embeddings.
    centroid_path = utils.path_to_hash(image_path) + ".pkl"
    centroids_pkl = str(cache_dir / Path(centroid_path))
    dup_centroid = _compute_dup_centroid(embeddings_info, centroids_pkl)

    logger.info(
        f"Converted images to embeddings and computed centroids. Now cleaning.."
    )
    cleaned_embeddings = _clean_embeddings(hash_set, dup_centroid)

    return cleaned_embeddings


def _clean_embeddings(
    embedding_file_hashes: set,
    dup_centroid: dict,
) -> list:
    """
    Clean the embeddings by removing those close to duplicate centroids.

    This function iteratively opens each embedding pickle file, reads the embeddings,
    compares each embedding against the centroids of significant clusters (duplicates),
    and selects the embedding that is furthest away from all centroids.

    If a pickle file contains only one embedding, it is selected without comparison.
    The output is a list of cleaned embeddings, ideally one per image.

    Args:
        embedding_file_hashes (set): Set of hash strings corresponding to embedding pickle files.
        dup_centroid (dict): Dictionary mapping cluster labels to centroid embeddings of significant clusters.

    Returns:
        list: List of cleaned embedding dictionaries.
    """
    cleaned_embeddings = []

    # Convert centroids to a list for easier computation
    centroid_embeddings = list(dup_centroid.values())

    for pickle_cache in embedding_file_hashes:
        if not os.path.exists(pickle_cache):
            logger.error(f"Error, embedding file not found: {pickle_cache}")
            continue

        with open(pickle_cache, "rb") as f:
            embeddings = pickle.load(f)

        if not embeddings:
            # No embeddings in this file
            continue

        if len(embeddings) == 1:
            # Only one embedding, select it without comparison
            cleaned_embeddings.append(embeddings[0])
        else:
            # Multiple embeddings, select the one furthest from all centroids
            max_distance = -np.inf
            selected_embedding_info = None

            for embedding_info in embeddings:
                embedding = embedding_info["embedding"]
                # Calculate distances to each centroid
                distances = (
                    [
                        np.linalg.norm(embedding - centroid)
                        for centroid in centroid_embeddings
                    ]
                    if centroid_embeddings
                    else [0]
                )

                # Use the minimum distance to any centroid
                min_distance_to_centroids = min(distances)

                # Select the embedding with the maximum of these minimum distances
                if min_distance_to_centroids > max_distance:
                    max_distance = min_distance_to_centroids
                    selected_embedding_info = embedding_info

            if selected_embedding_info is not None:
                cleaned_embeddings.append(selected_embedding_info)
            else:
                # If no embedding is selected, you may decide to handle this case differently
                pass  # For now, we do nothing

    return cleaned_embeddings


def _compute_dup_centroid(
    embeddings_info: list,
    centroid_cache_path: str,
    min_cluster_size: int = 2,
    significance_multiplier: int = 3,
) -> dict:
    """Preprocess the embeddings to identify recurring individuals (e.g., professors).

    Clusters the embeddings, identifies significant clusters (individuals appearing in many images),
    computes the centroids of these clusters, and saves them to the specified cache path.

    Args:
        embeddings_info (list): List of dictionaries containing embeddings and associated metadata.
        centroid_cache_path (str): Path to save the centroids of significant clusters.

    Returns:
        dict: Dictionary mapping cluster labels to centroid embeddings of significant clusters.
    """

    # Check if centroid cache exists
    if os.path.exists(centroid_cache_path):
        logger.info(f"Loading centroids from cache: {centroid_cache_path}")
        with open(centroid_cache_path, "rb") as f:
            centroids = pickle.load(f)
        return centroids

    # Extract embeddings from embeddings_info
    embeddings = np.array([item["embedding"] for item in embeddings_info])
    embeddings = np.squeeze(embeddings, axis=1)  # Remove extra dimension if present
    cluster_obj = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, leaf_size=50)
    labels = cluster_obj.fit_predict(embeddings)

    # Assign cluster labels back to embeddings_info
    for idx, label in enumerate(labels):
        embeddings_info[idx]["cluster_label"] = label

    # Count the number of embeddings in each cluster
    cluster_counts = Counter(labels)
    dup_threshold = min_cluster_size * significance_multiplier

    dup_clusters = [
        label
        for label, count in cluster_counts.items()
        if count >= dup_threshold and label != -1  # Exclude noise label (-1)
    ]

    logger.info(
        f"There are {len(dup_clusters)} significant clusters that has {dup_threshold} + embeddings."
    )

    # Compute centroids of significant clusters
    centroids = {}
    for cluster_label in dup_clusters:
        # Get embeddings belonging to this cluster
        cluster_embeddings = np.array(
            [
                item["embedding"]
                for item in embeddings_info
                if item["cluster_label"] == cluster_label
            ]
        )
        # Compute the centroid (mean embedding)
        centroid = np.mean(cluster_embeddings, axis=0)
        centroids[cluster_label] = centroid

    # Save centroids to cache
    with open(centroid_cache_path, "wb") as f:
        pickle.dump(centroids, f)

    return centroids


def run_dp2_pipeline(source_path, target_path, debug=True):
    """Run the entire DP2 pipeline."""

    # run preprocess first
    source_images = search_all_images(source_path)
    target_images = search_all_images(target_path)
    all_images = source_images + target_images

    cleaned_embeddings = _preprocess_embeddings(all_images)
    logger.info(f"Source images: {len(source_images)} + Target images: {len(target_images)} = {len(all_images)}")
    logger.info(f"Cleaned embeddings: {len(cleaned_embeddings)}")

    if debug:
        # iterate over each path, and draw the mask. And save for sanity check.
        save_path_dir = './test' # save the images here for sanity check.
        os.makedirs(save_path_dir, exist_ok=True)
        embeddings_sanity_check(cleaned_embeddings, save_path_dir)

    
    IPython.embed()


if __name__ == "__main__":

    demo_path_a = str(Path(r"/Users/philipcho/for_phil/bcit_copy/a"))
    demo_path_b = str(Path(r"/Users/philipcho/for_phil/bcit_copy/b"))
    test = run_dp2_pipeline(demo_path_a, demo_path_b)
