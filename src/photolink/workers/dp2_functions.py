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
import cv2
from sklearn.cluster import DBSCAN
import numpy as np
from collections import Counter
import hdbscan
from sklearn.cluster import MeanShift


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


def visualize_embeddings(embeddings_info: List[Dict], output_dir: str = "debug"):
    """Visualize embeddings by drawing bounding boxes on images.

    For each data entry in embeddings_info, load the image, draw the bounding box,
    assert that the length of 'bbox' is 1, and save it to the specified output directory
    with the basename of the image.

    Args:
        embeddings_info (List[Dict]): List of embedding dictionaries.
        output_dir (str): Directory to save the visualized images.
    """
    os.makedirs(output_dir, exist_ok=True)

    for data in embeddings_info:
        image_path = data['image_path']
        bboxes = data['bbox']

        # Assert that the length of 'bbox' is 1
        if not len(bboxes) == 1:
            logger.error(f"Expected one bounding box, but got {len(bboxes)} for image {image_path}")
            continue

        # Load the image
        img = safe_load_image(image_path)

        # Draw the bounding box
        for box in bboxes:
            x1, y1, x2, y2, *rest = box  # Unpack the bounding box coordinates
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Draw the bounding box on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

        # Save the image to the output directory with the basename of the image
        basename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, basename)

        # Convert image to BGR format if it's in RGB (since OpenCV uses BGR)
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imwrite(output_path, img)
        logger.info(f"Saved visualized image to {output_path}")


def _cluster_clean_embeddings(embeddings_info: List[Dict]) -> List[Dict]:
    """Cluster and clean embeddings, only when flagged."""
    # Collect all combined embeddings from flagged data entries
    combined_embeddings = []
    combined_embeddings_indices = []  # To track the source of each embedding
    flag_count = 0
    for data_idx, data in enumerate(embeddings_info):
        if data.get('flagged', True):  # Process entries where flagged == True
            flag_count += 1

            # For each instance (bounding box)
            for inst_idx in range(len(data['bbox'])):
                reid_emb = data['reid_embedding'][inst_idx]  # Shape (2048,)
                face_embs = data['face_embedding'][inst_idx]  # Could be multiple embeddings

                # Ensure face_embs is an array of embeddings
                if isinstance(face_embs, np.ndarray) and face_embs.ndim == 2:
                    # Multiple face embeddings for this instance
                    for face_emb in face_embs:
                        # Merge embeddings
                        combined_emb = np.concatenate([reid_emb, face_emb])
                        combined_embeddings.append(combined_emb)
                        combined_embeddings_indices.append((data_idx, inst_idx))
                elif isinstance(face_embs, np.ndarray) and face_embs.ndim == 1:
                    # Single face embedding
                    combined_emb = np.concatenate([reid_emb, face_embs])
                    combined_embeddings.append(combined_emb)
                    combined_embeddings_indices.append((data_idx, inst_idx))
                else:
                    logger.warning(f"Unexpected face embedding shape for data index {data_idx}")
                    continue

    logger.info(f"Found {flag_count}/{len(embeddings_info)} flagged data entries with multiple embeddings.")

    # No embeddings to cluster
    if not combined_embeddings:
        return embeddings_info

    logger.info(f"Clustering {len(combined_embeddings)} combined embeddings.")

    combined_embeddings = np.array(combined_embeddings)

    # clustering = DBSCAN(eps=0.2, min_samples=3, metric='cosine').fit(combined_embeddings)
    # labels = clustering.labels_
    clusterer = hdbscan.HDBSCAN(min_cluster_size=4, metric='euclidean')
    labels = clusterer.fit_predict(combined_embeddings)

    # Identify the biggest cluster (excluding noise points with label -1)
    label_counts = Counter(labels[labels != -1])
    if not label_counts:
        # No clusters found
        return embeddings_info

    biggest_cluster_label = label_counts.most_common(1)[0][0]

    # Build a mapping from (data_idx, inst_idx) to cluster labels
    label_mapping = {}
    for idx, ((data_idx, inst_idx), label) in enumerate(zip(combined_embeddings_indices, labels)):
        label_mapping.setdefault(data_idx, {})
        label_mapping[data_idx][inst_idx] = label

    # Now, for each flagged data entry, clean the embeddings
    for data_idx, data in enumerate(embeddings_info):
        if data.get('flagged', True):
            cleaned_bbox = []
            cleaned_reid_embedding = []
            cleaned_face_embedding = []
            instances_in_biggest_cluster = []
            instances_not_in_biggest_cluster = []
            for inst_idx in range(len(data['bbox'])):
                label = label_mapping.get(data_idx, {}).get(inst_idx, -2)  # Default to -2 if not found
                # Check if the instance is in the biggest cluster
                if label == biggest_cluster_label:
                    instances_in_biggest_cluster.append(inst_idx)
                else:
                    instances_not_in_biggest_cluster.append(inst_idx)

            # If there are instances not in the biggest cluster, keep them
            for inst_idx in instances_not_in_biggest_cluster:
                cleaned_bbox.append(data['bbox'][inst_idx])
                cleaned_reid_embedding.append(data['reid_embedding'][inst_idx])
                cleaned_face_embedding.append(data['face_embedding'][inst_idx])

            # If no instances not in the biggest cluster, keep one instance from the biggest cluster
            if not cleaned_bbox and instances_in_biggest_cluster:
                logger.info(f"All instances in data index {data['image_path']} are in the biggest cluster, keeping one instance.")
                inst_idx = instances_in_biggest_cluster[0]  # Keep the first instance
                cleaned_bbox.append(data['bbox'][inst_idx])
                cleaned_reid_embedding.append(data['reid_embedding'][inst_idx])
                cleaned_face_embedding.append(data['face_embedding'][inst_idx])

            # Update data with cleaned lists
            data['bbox'] = cleaned_bbox
            data['reid_embedding'] = cleaned_reid_embedding
            data['face_embedding'] = cleaned_face_embedding
            data['flagged'] = len(data['bbox']) > 1  # Update flagged status

    return embeddings_info

def validate_embeddings(embeddings_info: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Validate embeddings, separating entries with expected bounding boxes != 1.

    Args:
        embeddings_info (List[Dict]): List of embedding dictionaries.

    Returns:
        valid_embeddings (List[Dict]): Entries with exactly one bounding box.
        failed_embeddings (List[Dict]): Entries with zero or multiple bounding boxes.
    """
    valid_embeddings = []
    failed_embeddings = []
    for data in embeddings_info:
        num_bboxes = len(data['bbox'])

        # TODO: Call Kosmos here, and then update the data on uncertain ones. 
        if num_bboxes != 1:
            logger.warning(f"Data entry {data['image_path']} has {num_bboxes} bounding boxes, expected 1.")
            failed_embeddings.append(data)
        else:
            valid_embeddings.append(data)
    return valid_embeddings, failed_embeddings



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
            data['flagged'] = False

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

            if len(data["bbox"]) != len(data["face_embedding"]) or len(data["bbox"]) != len(data["reid_embedding"]):
                raise ValueError("Length of bbox, face_embedding and reid_embedding should be same.")

            # when more than one face_embedding, bbox or reid_embedding is present, flag it.
            if len(data["bbox"]) > 1 or len(data["face_embedding"]) > 1 or len(data["reid_embedding"]) > 1:
                data['flagged'] = True

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

    #clean up
    source_embeddings_info = _cluster_clean_embeddings(source_embeddings_info)
    source_embeddings_info, failed_source_info = validate_embeddings(source_embeddings_info)
    logger.info(f"Source embeddings cleaned. {len(source_embeddings_info)} embeddings remaining, {len(failed_source_info)} failed.")
    visualize_embeddings(source_embeddings_info)

    # Compute embeddings for reference images
    # reference_embeddings_info, reference_hash_set = _precompute_embeddings(
    #     reference_images, debug
    # )
    # logger.info(f"Reference embeddings computed for {len(reference_images)} images.")

    # IPython.embed()


if __name__ == "__main__":
    print("Start to run the code")

    # on stage images
    source_images = search_all_images(Path("~/for_phil/bcit_copy/a").expanduser())

    # off stage images
    reference_images = search_all_images(Path("~/for_phil/bcit_copy/b").expanduser())

    # run pipeline
    run(source_images, reference_images, debug=False)
