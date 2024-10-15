"""Add lower level dp2 functions to avoid cluttering the main functional modules."""

import os
from pathlib import Path

from loguru import logger

from photolink import get_application_path, get_config
from photolink.utils.function import safe_load_image, search_all_images
from photolink.pipeline import get_cache_dir
from photolink.models.yolo_seg import get_segmentation
from photolink.models.fastreid import isolate_instance, get_reid_embedding
import pickle
import numpy as np
import hdbscan
import photolink.utils.function as utils
from collections import Counter


def compute_human_embeddings(image_path: str) -> dict:
    """Use yoloworld to detect humans based on precomputed embeddings.

    Iterate over list of images, and return dict based on the image name as key.
    """
    output_dict = {}

    if not os.path.isdir(image_path):
        logger.error(f"Image path does not exist or is not dir: {image_path}")
        output_dict["error"] = f"Image path does not exist: {image_path}"
        return output_dict

    source_images = search_all_images(image_path)

    if len(source_images) == 0:
        logger.error(f"Empty image list: {source_images}")
        output_dict["error"] = f"Empty image list: {source_images}"
        return output_dict

    config = get_config()

    # store all he embeddings here.
    cache_dir = get_cache_dir() / Path(config.get("FASTREID", "EMBEDDING_CACHE_DIR"))
    cache_dir.mkdir(exist_ok=True)

    # start iterating over the images.
    embeddings_info = []

    for idx, img_path in enumerate(source_images):

        pickle_cache = cache_dir / Path(f"{Path(img_path).stem}.pkl")

        # if the pickle cache exists, load it and continue.
        if pickle_cache.exists():
            with open(pickle_cache, "rb") as f:
                embeddings = pickle.load(f)

                # check if the embeddings have correct keys.
                if not all(
                    k in embeddings[0]
                    for k in [
                        "embedding",
                        "image_path",
                        "box_index",
                        "box",
                        "confidence",
                        "class",
                        "mask",
                    ]
                ):
                    logger.error(f"Invalid keys in pickle cache: {pickle_cache}")
                    error = f"Invalid keys in pickle cache: {pickle_cache}"
                    output_dict["error"] = error
                    return output_dict

                embeddings_info.append(embeddings)
                logger.info(f"Loaded embeddings from cache: {pickle_cache}")
                continue

        else:
            try:
                img = safe_load_image(img_path)
                boxes, segments, masks = get_segmentation(img_path)
            except Exception as e:
                logger.error(f"Error processing image: {img_path}, {e}")
                continue

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

                # dump the embeddings to the cache.
                with open(pickle_cache, "wb") as f:
                    pickle.dump(embeddings_info, f)
                
                logger.info(f"Saved embeddings to cache: {pickle_cache}")

    # run clustering on the embeddings.
    centroid_path = utils.path_to_hash(image_path) + ".pkl"
    centroids_pkl = str(cache_dir / Path(centroid_path))
    dup_centroid = identify_duplicate(embeddings_info, centroids_pkl)

    return output_dict


def identify_duplicate(
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


if __name__ == "__main__":

    demo_path = str(Path(r"/Users/philipcho/for_phil/bcit_copy"))
    test = compute_human_embeddings(demo_path)
