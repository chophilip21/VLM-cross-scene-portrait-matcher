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

# set glboal variables
config = get_config()
cache_dir = get_cache_dir() / Path(config["FASTREID"]["EMBEDDING_CACHE_DIR"])

def _compute_dup_centroid(
    embeddings_info: List[Dict],
    centroid_cache_path: str,
    max_clusters: int = 10,
    debug: bool = False,
) -> Dict[int, np.ndarray]:
    """
    Compute duplicate centroids using Spectral Clustering on embeddings directly.
    """
    # Check if centroid cache exists
    if os.path.exists(centroid_cache_path) and not debug:
        logger.info(f"Loading centroids from cache: {centroid_cache_path}")
        with open(centroid_cache_path, "rb") as f:
            centroids = pickle.load(f)
        return centroids

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

    return centroids


def _clean_embeddings(
    hash_set: set,
    dup_centroid: Dict[int, np.ndarray],
    similarity_threshold: float = 0.9,
) -> List[Dict]:
    """
    Remove duplicate embeddings using the computed centroids and assign cluster labels to final candidates.

    Args:
        hash_set (set): Set of paths to cached embeddings (per image).
        dup_centroid (dict): Dictionary of centroid embeddings of recurring individuals.
        similarity_threshold (float): Threshold for excluding embeddings similar to centroids.

    Returns:
        list: List of cleaned embedding dictionaries, one per image, with assigned cluster labels.
    """
    cleaned_embeddings = []

    # Load all embeddings and group them by image path
    embeddings_by_image = {}
    for pickle_cache in hash_set:
        with open(pickle_cache, "rb") as f:
            embeddings = pickle.load(f)
            for embedding_info in embeddings:
                img_path = embedding_info["image_path"]
                embeddings_by_image.setdefault(img_path, []).append(embedding_info)

    centroid_embeddings = list(dup_centroid.values())
    centroid_labels = list(dup_centroid.keys())

    # For each image, select the best embedding
    for img_path, embedding_infos in embeddings_by_image.items():
        valid_embeddings = []
        for embedding_info in embedding_infos:
            # Compute cosine similarity with all centroids
            embedding = embedding_info["embedding"]
            embedding = np.squeeze(embedding, axis=0)
            if len(centroid_embeddings) > 0:
                similarities = cosine_similarity([embedding], centroid_embeddings)[0]
                max_sim_idx = np.argmax(similarities)
                max_sim = similarities[max_sim_idx]
                assigned_cluster_label = centroid_labels[max_sim_idx]
            else:
                max_sim = 0  # No centroids to compare with
                assigned_cluster_label = -1  # Assign -1 when no centroids

            # Assign cluster label based on similarity
            if max_sim >= similarity_threshold:
                # Similar to a centroid (recurring individual)
                embedding_info["cluster_label"] = assigned_cluster_label
            else:
                # Not similar to any centroid
                embedding_info["cluster_label"] = -1  # Assign -1 for unique individuals

            # Collect all embeddings, regardless of similarity, to select the best one later
            valid_embeddings.append((embedding_info, max_sim))

        # Select the best embedding per image
        # Prefer embeddings not similar to centroids (cluster_label == -1)
        # If multiple embeddings have cluster_label == -1, select one with lowest max_sim (least similar to centroids)
        # If all embeddings have cluster_label != -1, select one with lowest max_sim (most similar to centroids)

        # First, filter embeddings with cluster_label == -1
        non_recurring_embeddings = [ve for ve in valid_embeddings if ve[0]["cluster_label"] == -1]

        if non_recurring_embeddings:
            # Select among embeddings not similar to centroids
            selected_embedding_info = min(non_recurring_embeddings, key=lambda x: x[1])[0]
        else:
            # All embeddings are similar to centroids; select the one most similar (lowest max_sim)
            selected_embedding_info = min(valid_embeddings, key=lambda x: x[1])[0]

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

    # Clean embeddings using the updated embeddings_info
    src_centroids_pkl_path = str(cache_dir / Path(utils.path_to_hash("".join(source_images)) + ".pkl"))
    src_dup_centroid = _compute_dup_centroid(
        src_embeddings_info,
        src_centroids_pkl_path,
        debug=debug,
    )

    src_cleaned_embeddings = _clean_embeddings(src_hash_set, src_dup_centroid)
    logger.info(f"Cleaned embeddings: {len(src_cleaned_embeddings)}")

    if debug:
        # Save images for sanity check
        save_path_dir = "./test"
        if os.path.exists(save_path_dir):
            shutil.rmtree(save_path_dir)
        os.makedirs(save_path_dir, exist_ok=True)
        
        # save individual images and tsne
        embeddings_sanity_check(src_cleaned_embeddings, save_path_dir)
        visualize_embeddings_tsne_interactive(src_cleaned_embeddings, save_path_dir)
        



if __name__ == "__main__":

    demo_path_a = str(Path(r"/Users/philipcho/for_phil/bcit_copy/a"))
    demo_path_b = str(Path(r"/Users/philipcho/for_phil/bcit_copy/b"))
    test = run_dp2_pipeline(demo_path_a, demo_path_b)
