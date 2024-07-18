"""Lowest functional layer for Various ML algorithms/functions (face detection and recognition, and clustering)."""

import photolink.models.yunet as yunet
import photolink.models.sface as sface
import photolink.utils.enums as enums
import photolink.utils.function as function
import os
import shutil
import multiprocessing as mp
import numpy as np
import faiss
import hdbscan
import cv2
from pathlib import Path
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from loguru import logger
import IPython


# Global variables for pre-loaded models
YUNET_MODEL = yunet.load_model()
SFACE_MODEL = sface.load_model()


def _run_ml_model(
    image_path: str, save_path: Path, fail_path: Path, keep_top_n: int = 3
):
    """Process face detection and recognition for images, save embeddings to save_path as xz file.

    Instead of converting all faces to embeddings, only largest top n faces are converted. Result should have both aligned face and converted embeddings as keys.
    """

    # Use the checksum of the image as the hash to avoid duplicates.
    image_hash = function.checksum(image_path)
    save_embedding_name = save_path / Path(image_hash + ".xz")

    # no need to calculate if the embeddings already exist.
    if save_embedding_name.exists():
        logger.info(f"Embeddings already exists for {image_path}. Skipping processing.")
        return (image_path, image_hash)

    # Use the pre-loaded global models
    global YUNET_MODEL, SFACE_MODEL
    detection_result = YUNET_MODEL.run_face_detection(image_path)
    failed_image = fail_path / Path(os.path.basename(image_path))

    if "error" in detection_result:
        shutil.copy(image_path, failed_image)
        warning = f"Warning! Face detection error on source image {image_path}: {detection_result['error']}"

        return warning

    if "faces" not in detection_result:
        shutil.copy(image_path, failed_image)
        warning = f"Warning! Face not detected on source image {image_path}. Faces probably not there, or too small."

        return warning

    image = detection_result["image"]
    faces = detection_result["faces"]

    # Only go over the top n faces instead of going for all faces.
    if faces.shape[0] < keep_top_n:
        keep_top_n = faces.shape[0]

    # bb = [x, y, w, h, landmarks]
    faces = sorted(faces, key=lambda face: face[2] * face[3], reverse=True)[:keep_top_n]
    embedding_dict = SFACE_MODEL.run_embedding_conversion(image, faces)

    # add image path to the dictionary.
    embedding_dict["image_path"] = image_path

    if "error" in embedding_dict:
        shutil.copy(image_path, failed_image)
        warning = f"Warning! Face recognition error on source image {image_path}: {embedding_dict['error']}"

        return warning

    # pickle dump all face embeddings found in the image.
    function.compress_save(embedding_dict, save_embedding_name)

    # return image path and hash as result.
    return (image_path, image_hash)


def run_task(entry, stop_flag, save_path, fail_path, keep_top_n):
    """Wrapper function for running the ML model. Return None if stop flag is set."""
    if stop_flag.value:
        return None
    return _run_ml_model(entry, save_path=save_path, fail_path=fail_path)


def run_model_mp(
    entries,
    num_workers: int,
    save_path: str,
    fail_path: str,
    keep_top_n: int,
    stop_event: mp.Event,
):
    """Use concurrent.futures to run the models and save each result to a pickle file to cache path. Catch signals to stop the process."""

    # Using Manager for a shared flag
    manager = mp.Manager()
    stop_flag = manager.Value("i", 0)
    entries = sorted(entries)

    # This should have already been setup in the main function.
    hash_json_dict = function.read_hash_file(raise_missing_error=False)
    hash_json_dict_ = manager.dict(hash_json_dict)  # for thread safety.

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_entry = {
            executor.submit(
                run_task, entry, stop_flag, save_path, fail_path, keep_top_n
            ): entry
            for entry in entries
        }

        try:
            for future in as_completed(future_to_entry):
                if stop_event.is_set():
                    logger.warning(
                        "JobProcessor: Stop event detected, shutting down executor."
                    )
                    stop_flag.value = 1  # Set the stop flag
                    executor.shutdown(wait=False)
                    break
                
                try:
                    result = future.result()

                    if isinstance(result, str):
                        logger.warning(result)
                        continue

                    # result is a tuple of (image_path, image_hash). Add to dict.
                    image_path, image_hash = future.result()
                    hash_json_dict_[image_hash] = image_path
                    logger.info(
                        f"run_model_mp: Processed entry {image_path}, hash: {image_hash}"
                    )
                except Exception as e:
                    entry = future_to_entry[future]
                    logger.error(f"Error processing entry {entry}: {e}")

        except Exception as e:
            logger.error(f"Error during concurrent processing: {e}")
            raise e

    # save the hash json file.
    # IPython.embed()
    function.write_hash_file(hash_json_dict_)


def read_embedding_file(embedding_path) -> dict:
    """Read and validate embeddings file to retrieve dictionary from pickle path."""

    data = {}

    embedding_dict = function.decompress_load(embedding_path)

    if "embeddings" not in embedding_dict:
        raise ValueError(
            f"Failed to read embeddings. Key for embeddings not found in {embedding_path}"
        )

    if "faces" not in embedding_dict:
        raise ValueError(
            f"Failed to read embeddings. Key for faces not found in {embedding_path}"
        )

    embeddings = np.array(embedding_dict["embeddings"])
    if embeddings.shape[1] != 128:
        raise ValueError(
            f"Error on {embedding_path}. Embedding must be a (N, 128) numpy array, not {embeddings.shape}"
        )

    data["faces"] = np.array(embedding_dict["faces"])
    data["embeddings"] = embeddings

    return data


def match_embeddings(
    source_cache: Path,
    reference_cache: Path,
    output_path: Path,
    stop_event: mp.Event,
) -> dict:
    """Match the embeddings and save the match. Return result status."""

    # retreive the lookup table. This must exist at this point.
    hash_json_dict = function.read_hash_file(raise_missing_error=True)

    # Get a list of relevant source/reference embeddings.
    source_embeddings = function.get_relevant_embeddings(source_cache, "source")
    reference_embeddings = function.get_relevant_embeddings(
        reference_cache, "reference"
    )

    result = {}
    faiss_index = faiss.IndexFlatL2(128)

    # start by reading each embedding file, and adding to faiss index.
    for i, embedding_file in enumerate(source_embeddings):

        if stop_event.is_set():
            result["error"] = "Stop event detected. Exiting matching."
            return result

        progress = math.ceil(50 + ((i + 1) / len(source_embeddings) * 25))

        if progress % 5 == 0:
            logger.info(f"Post-Processing:{int(progress)}")

        try:
            source_embedding_dict = read_embedding_file(
                source_cache / Path(embedding_file)
            )

            source_embedding = source_embedding_dict["embeddings"]

            for face_embedding in source_embedding:

                # check number of dimensions. Should be (1, 128)
                if face_embedding.ndim == 1:
                    face_embedding = np.expand_dims(face_embedding, axis=0)

                faiss_index.add(face_embedding)

        except Exception as e:
            result["error"] = f"Error adding source embeddings to faiss index: {e}"
            return result

    for i, file in enumerate(reference_embeddings):

        progress = math.ceil(75 + ((i + 1) / len(reference_embeddings) * 25))

        if progress % 5 == 0:
            logger.info(f"Post-Processing:{int(progress)}")

        try:
            reference_embedding_dict = read_embedding_file(reference_cache / Path(file))

            reference_embedding = reference_embedding_dict["embeddings"]

            for face_embedding in reference_embedding:

                # check number of dimensions. Should be (1, 128)
                if face_embedding.ndim == 1:
                    face_embedding = np.expand_dims(face_embedding, axis=0)

                # search for the nearest embedding in the source embeddings.
                D, I = faiss_index.search(face_embedding, 1)

                predicted_label = Path(source_embeddings[I[0][0]]).stem

                if predicted_label not in hash_json_dict:
                    raise ValueError(
                        f"Predicted label {predicted_label} not found in source_dict."
                    )

                predicted_source_input_path = hash_json_dict[predicted_label]

                # now get the equivalent reference image.
                reference_label = Path(file).stem

                if reference_label not in hash_json_dict:
                    raise ValueError(
                        f"Reference label {reference_label} not found in reference_dict."
                    )

                ref_img_input_path = hash_json_dict[reference_label]

                # to output folder, create a folder based predicted_label.
                predicted_label_folder = output_path / Path(predicted_label)
                predicted_label_folder.mkdir(parents=True, exist_ok=True)

                # copy everything to the output folder.
                predicted_source_output_path = predicted_label_folder / Path(
                    os.path.basename(predicted_source_input_path),
                )

                shutil.copy(predicted_source_input_path, predicted_source_output_path)

                ref_img_output_path = predicted_label_folder / Path(
                    os.path.basename(ref_img_input_path)
                )

                shutil.copy(ref_img_input_path, ref_img_output_path)
                result["status"] = "success"

        except Exception as e:
            result["error"] = f"Error matching embeddings: {e}"
            return result

    return result


def cluster_embeddings(
    source_cache: Path,
    clustering_algorithm: str,
    eps: float,
    min_samples: int,
    output_path: Path,
    fail_path: Path,
    stop_event: mp.Event,
) -> dict:
    """Cluster embeddings, and save the output. Return result status."""

    result = {}
    result["missed_count"] = 0

    # get list of relevant embeddings.
    source_embeddings = function.get_relevant_embeddings(source_cache, "source")
    hash_json_dict = function.read_hash_file(raise_missing_error=True)

    loaded_embeddings = []
    loaded_faces = []
    cluster_obj = None

    # init clustering algorithm.
    if clustering_algorithm == enums.ClusteringAlgorithm.DBSCAN.value:
        raise NotImplementedError(
            f"Clustering algorithm {clustering_algorithm} not supported."
        )
    elif clustering_algorithm == enums.ClusteringAlgorithm.OPTICS.value:
        raise NotImplementedError(
            f"Clustering algorithm {clustering_algorithm} not supported."
        )
    elif clustering_algorithm == enums.ClusteringAlgorithm.HDBSCAN.value:
        cluster_obj = hdbscan.HDBSCAN(min_cluster_size=min_samples, leaf_size=50)
    else:
        raise NotImplementedError(
            f"Clustering algorithm {clustering_algorithm} not supported."
        )

    # because there are multiple faces, we need to keep track of the embeddings.
    face_to_embedding_file_table = {}
    index_count = 0
    for file in source_embeddings:

        if stop_event.is_set():
            return

        try:
            embedding_dict = read_embedding_file(source_cache / Path(file))
            embedding = embedding_dict["embeddings"]
            aligned_face = embedding_dict["faces"]

        except Exception as e:
            result["error"] = f"Error reading embeddings file when clustering: {e}"
            return result

        # append the embeddings and corresponding faces to a list for clustering.
        for i, face_embedding in enumerate(embedding):
            loaded_embeddings.append(face_embedding.flatten())
            loaded_faces.append(aligned_face[i])

            # keep track of the face to embedding file with face index.
            face_to_embedding_file_table[index_count] = file
            index_count += 1

    embeddings = np.array(loaded_embeddings)

    try:
        labels = cluster_obj.fit_predict(embeddings)
    except Exception as e:
        result["error"] = f"Error during cluster fit_predict embeddings: {e}"
        return result

    # now save the output.
    for i, label in enumerate(labels):

        if stop_event.is_set():
            return

        progress = math.ceil(50 + ((i + 1) / len(labels) * 50))
        if progress % 5 == 0:
            logger.info(f"Post-Processing:{int(progress)}")

        # find the original image from look up table.
        backtracked_file_name = Path(face_to_embedding_file_table[i])

        # assert the hash is correct length.
        if len(backtracked_file_name.stem) != 64:
            logger.warning(
                f"Invalid hash length for {backtracked_file_name.stem}. Skipping."
            )
            continue

        source_img_path = hash_json_dict[backtracked_file_name.stem]

        # label -1 indicates failed clustering.
        if label == -1:
            failed_img_output_path = str(
                fail_path / Path(os.path.basename(source_img_path))
            )
            shutil.copy(source_img_path, failed_img_output_path)
            logger.warning(
                f"Failed clustering on {source_img_path}. Saved to {failed_img_output_path}"
            )
            result["missed_count"] += 1
            continue

        label_folder = output_path / Path("subject-no-" + str(label))
        label_folder.mkdir(parents=True, exist_ok=True)

        source_img_output_path = str(
            label_folder / Path(os.path.basename(source_img_path))
        )

        shutil.copy(source_img_path, source_img_output_path)

        # save face sample for sanity check.
        try:
            aligned_face_sample_output_path = label_folder / Path("target_face.jpg")
            cv2.imwrite(aligned_face_sample_output_path, loaded_faces[i])
        except Exception as e:
            result["error"] = f"Error saving aligned face sample: {e}"
            return result

    result["status"] = "success"

    return result
