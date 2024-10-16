"""Add lower level dp2 functions to avoid cluttering the main functional modules."""

import os
import pickle
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Union

import IPython
import networkx as nx
import numpy as np
from loguru import logger
from scipy.sparse import csgraph
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

import photolink.utils.function as utils
from photolink import get_config
from photolink.models.fastreid import get_reid_embedding, isolate_instance
from photolink.models.yolo_seg import get_segmentation
from photolink.pipeline import get_cache_dir
from photolink.utils.function import safe_load_image, search_all_images
from photolink.workers.draw import embeddings_sanity_check, visualize_embeddings_tsne_interactive
from sklearn.metrics import silhouette_score

def _compute_dup_centroid(
    embeddings_info: List[Dict],
    centroid_cache_path: str,
    max_clusters: int = 10,
    debug: bool = False,
) -> Tuple[Dict[int, np.ndarray], List[Dict]]:
    """
    Compute duplicate centroids using Spectral Clustering on embeddings directly
    and assign cluster labels.
    """
    # Check if centroid cache exists
    if os.path.exists(centroid_cache_path) and not debug:
        logger.info(f"Loading centroids from cache: {centroid_cache_path}")
        with open(centroid_cache_path, "rb") as f:
            centroids = pickle.load(f)
        return centroids, embeddings_info  # Return embeddings_info as is

    # Extract embeddings (do not normalize)
    embeddings = np.array([item["embedding"] for item in embeddings_info])
    embeddings = np.squeeze(embeddings, axis=1)

    # Determine the optimal number of clusters using Silhouette Score
    best_score = -1
    best_n_clusters = 2
    best_labels = None

    for n_clusters in range(2, min(len(embeddings), max_clusters + 1)):
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity='rbf',  # Uses RBF kernel to compute affinity matrix
            gamma=1.0,       # Adjust gamma as needed
            assign_labels='kmeans',
            random_state=42
        )
        labels = spectral.fit_predict(embeddings)

        # Check if the clustering result is valid
        if len(set(labels)) < 2:
            continue

        # Compute the Silhouette Score
        try:
            score = silhouette_score(embeddings, labels, metric='euclidean')
            logger.debug(f"Silhouette Score for n_clusters={n_clusters}: {score}")
        except Exception as e:
            logger.warning(f"Silhouette Score computation failed for n_clusters={n_clusters}: {e}")
            continue

        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters
            best_labels = labels

    if best_labels is None:
        # Fallback if no suitable clustering found
        logger.info("No suitable clustering found using Silhouette Score. Using n_clusters=2")
        spectral = SpectralClustering(
            n_clusters=2,
            affinity='rbf',
            gamma=1.0,
            assign_labels='kmeans',
            random_state=42
        )
        labels = spectral.fit_predict(embeddings)
    else:
        labels = best_labels
        logger.info(f"Best n_clusters according to Silhouette Score: {best_n_clusters} with score {best_score}")

    # Assign cluster labels back to embeddings_info
    for idx, label in enumerate(labels):
        embeddings_info[idx]["cluster_label"] = label

    # Compute centroids of clusters
    centroids = {}
    unique_labels = np.unique(labels)
    for cluster_label in unique_labels:
        # Get embeddings belonging to this cluster
        cluster_embeddings = embeddings[labels == cluster_label]
        # Compute the centroid (mean embedding)
        centroid = np.mean(cluster_embeddings, axis=0)
        centroids[cluster_label] = centroid

    # Save centroids to cache
    with open(centroid_cache_path, "wb") as f:
        pickle.dump(centroids, f)

    return centroids, embeddings_info


def _clean_embeddings(
    embeddings_info: List[Dict],
    dup_centroid: Dict[int, np.ndarray],
    distance_threshold: float = 0.5,  # Adjust based on your data
) -> List[Dict]:
    """
    Clean the embeddings by removing those close to duplicate centroids.

    Args:
        embeddings_info (list): List of embeddings with cluster labels.
        dup_centroid (dict): Dictionary of centroid embeddings of clusters.
        distance_threshold (float): Threshold for excluding embeddings close to centroids.

    Returns:
        list: List of cleaned embedding dictionaries.
    """
    cleaned_embeddings = []

    centroid_embeddings = list(dup_centroid.values())
    centroid_embeddings = np.array(centroid_embeddings)

    # Group embeddings by image
    embeddings_by_image = {}
    for embedding_info in embeddings_info:
        img_path = embedding_info["image_path"]
        if img_path not in embeddings_by_image:
            embeddings_by_image[img_path] = []
        embeddings_by_image[img_path].append(embedding_info)

    for img_path, embeddings_list in embeddings_by_image.items():
        if not embeddings_list:
            continue

        valid_embeddings = []

        for embedding_info in embeddings_list:
            embedding = embedding_info["embedding"]
            embedding = np.squeeze(embedding, axis=0)

            # Calculate distances to centroids
            distances = (
                np.linalg.norm(centroid_embeddings - embedding, axis=1)
                if len(centroid_embeddings) > 0
                else [np.inf]
            )
            min_distance_to_centroids = min(distances)

            if min_distance_to_centroids >= distance_threshold:
                valid_embeddings.append(embedding_info)

        if valid_embeddings:
            # Select embedding with lowest box_index among valid embeddings
            selected_embedding_info = min(
                valid_embeddings, key=lambda x: x["box_index"]
            )
            cleaned_embeddings.append(selected_embedding_info)
        else:
            # Select embedding with lowest box_index anyway
            selected_embedding_info = min(embeddings_list, key=lambda x: x["box_index"])
            cleaned_embeddings.append(selected_embedding_info)

    return cleaned_embeddings


def _precompute_embeddings(
    image_path: Union[str, List[str]], debug: bool = False
) -> Tuple[Dict[int, np.ndarray], List[Dict], set]:
    """Compute embeddings and duplicate centroids, return centroids, embeddings_info, and hash_set."""
    if isinstance(image_path, str):
        source_images = search_all_images(image_path)
    else:
        source_images = image_path

    # Store all embeddings here
    config = get_config()
    cache_dir = get_cache_dir() / Path(config["FASTREID"]["EMBEDDING_CACHE_DIR"])
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

    # Compute centroids and get updated embeddings_info with cluster labels
    centroid_path = utils.path_to_hash("".join(source_images)) + ".pkl"
    centroids_pkl = str(cache_dir / Path(centroid_path))
    dup_centroid, embeddings_info = _compute_dup_centroid(
        embeddings_info,
        centroids_pkl,
        debug=debug,
    )

    logger.info(
        f"Converted images to embeddings and computed centroids. Now cleaning..."
    )

    return dup_centroid, embeddings_info, hash_set



def run_dp2_pipeline(source_path: str, target_path: str, debug: bool = True):
    """Run the entire DP2 pipeline."""
    # Run preprocess first
    source_images = search_all_images(source_path)
    target_images = search_all_images(target_path)
    all_images = source_images + target_images

    # Precompute embeddings and get embeddings_info with cluster labels
    dup_centroid, embeddings_info, hash_set = _precompute_embeddings(
        source_images, debug=debug
    )

    # Clean embeddings using the updated embeddings_info
    cleaned_embeddings = _clean_embeddings(embeddings_info, dup_centroid)

    logger.info(f"Cleaned embeddings: {len(cleaned_embeddings)}")

    if debug:
        # Save images for sanity check
        save_path_dir = "./test"
        if os.path.exists(save_path_dir):
            shutil.rmtree(save_path_dir)
        os.makedirs(save_path_dir, exist_ok=True)
        
        # save individual images
        embeddings_sanity_check(cleaned_embeddings, save_path_dir)

        # save tsne plot
        visualize_embeddings_tsne_interactive(cleaned_embeddings, save_path_dir)
        



if __name__ == "__main__":

    demo_path_a = str(Path(r"/Users/philipcho/for_phil/bcit_copy/a"))
    demo_path_b = str(Path(r"/Users/philipcho/for_phil/bcit_copy/b"))
    test = run_dp2_pipeline(demo_path_a, demo_path_b)
