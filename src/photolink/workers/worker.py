"""ML algorithms for face detection and recognition, and clustering."""

import photolink.models.yunet as yunet
import photolink.models.sface as sface
import photolink.utils.enums as enums
import os
import shutil
from functools import partial
import multiprocessing as mp
import pickle
import numpy as np
import faiss
import sklearn.cluster as c_algorithm
import hdbscan
import cv2
import math


def _run_ml_model(image_path: str, save_path: str, fail_path: str, keep_top_n: int = 3):
    """Process face detection and recognition for images, save embeddings to save_path as pkl file.

    Instead of converting all faces to embeddings, only largest top n faces are converted. Result should have both aligned face and converted embeddings as keys.
    """
    detection_result = yunet.run_face_detection(image_path)

    failed_image = os.path.join(fail_path, os.path.basename(image_path))

    if "error" in detection_result:
        shutil.copy(image_path, failed_image)
        warning = f"Face detection error on source image {image_path}: {detection_result['error']}"

        return warning

    if "faces" not in detection_result:
        shutil.copy(image_path, failed_image)
        warning = f"Face not detected on source image {image_path}. Faces probably not there, or too small."

        return warning

    image = detection_result["image"]
    faces = detection_result["faces"]

    # Only go over the top n faces instead of going for all faces.
    if faces.shape[0] < keep_top_n:
        keep_top_n = faces.shape[0]

    # bb = [x, y, w, h, landmarks]
    faces = sorted(faces, key=lambda face: face[2] * face[3], reverse=True)[:keep_top_n]
    embedding_dict = sface.run_embedding_conversion(image, faces)

    if "error" in embedding_dict:
        shutil.copy(image_path, failed_image)
        warning = f"Face recognition error on source image {image_path}: {embedding_dict['error']}"

        return warning

    save_embedding_name = os.path.join(
        save_path, os.path.basename(image_path).split(".")[0] + ".pkl"
    )

    # pickle dump all face embeddings found in the image.
    with open(save_embedding_name, "wb") as f:
        pickle.dump(embedding_dict, f)

    return True


def run_model_mp(
    entries,
    num_workers: int,
    chunksize: int,
    save_path: str,
    fail_path: str,
    keep_top_n: int = 3,
):
    """Use multiprocessing to run the models and each results to pickle file to cache path."""
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=num_workers)

    with pool as p:
        result = list(
            p.imap_unordered(
                partial(
                    _run_ml_model,
                    save_path=save_path,
                    fail_path=fail_path,
                    keep_top_n=keep_top_n,
                ),
                entries,
                chunksize=chunksize,
            )
        )

        if not result:
            print(f"Warning: {result}")


def read_embeddingpkl(embedding_path) -> dict:
    """Read and validate embeddings pkl file to retrieve dictionary from pickle path."""

    data = {}

    with open(embedding_path, "rb") as f:
        embedding_dict = pickle.load(f)

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

        data['faces'] = np.array(embedding_dict['faces'])
        data['embeddings'] = embeddings

    return data


def match_embeddings(
    source_cache: str,
    reference_cache: str,
    source_list_images: list,
    reference_list_images: list,
    output_path: str,
) -> dict:
    """Match the embeddings and save the match. Return result status."""

    result = {}

    faiss_index = faiss.IndexFlatL2(128)
    source_embeddings = [
        os.path.join(source_cache, file)
        for file in os.listdir(source_cache)
        if file.split(".")[-1] == "pkl"
    ]

    reference_embeddings = [
        os.path.join(reference_cache, file)
        for file in os.listdir(reference_cache)
        if file.split(".")[-1] == "pkl"
    ]

    # Create quick look up table for the source and reference images.
    source_dict = {
        file.split("/")[-1].split(".")[0]: file for file in source_list_images
    }
    reference_dict = {
        file.split("/")[-1].split(".")[0]: file for file in reference_list_images
    }

    # start by reading each embedding file, and adding to faiss index.
    for embedding_file in source_embeddings:
        try:
            source_embedding_dict = read_embeddingpkl(
                os.path.join(source_cache, embedding_file)
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

        progress = math.ceil(50 + ((i + 1) / len(reference_embeddings) * 50))

        if int(progress) % 2 == 0:
            print(f"POSTPROCESS_PROGRESS: {int(progress)}")

        try:
            reference_embedding_dict = read_embeddingpkl(
                os.path.join(reference_cache, file)
            )

            reference_embedding = reference_embedding_dict["embeddings"]

            for face_embedding in reference_embedding:

                # check number of dimensions. Should be (1, 128)
                if face_embedding.ndim == 1:
                    face_embedding = np.expand_dims(face_embedding, axis=0)

                # search for the nearest embedding in the source embeddings.
                D, I = faiss_index.search(face_embedding, 1)
                distance = D[0][0]

                # predicted label must exist in the lookup table.
                predicted_label = (
                    source_embeddings[I[0][0]].split("/")[-1].split(".")[0]
                )

                if predicted_label not in source_dict:
                    raise ValueError(
                        f"Predicted label {predicted_label} not found in source_dict."
                    )

                predicted_source_input_path = source_dict[predicted_label]

                # now get the equivalent reference image.
                reference_label = file.split("/")[-1].split(".")[0]

                if reference_label not in reference_dict:
                    raise ValueError(
                        f"Reference label {reference_label} not found in reference_dict."
                    )

                ref_img_input_path = reference_dict[reference_label]

                # to output folder, create a folder based predicted_label.
                predicted_label_folder = os.path.join(output_path, predicted_label)
                os.makedirs(predicted_label_folder, exist_ok=True)

                # copy everything to the output folder.
                predicted_source_output_path = os.path.join(
                    predicted_label_folder,
                    os.path.basename(predicted_source_input_path),
                )

                # print(f'Predicted source: {predicted_source_input_path} -> {predicted_source_output_path}')
                shutil.copy(predicted_source_input_path, predicted_source_output_path)

                # same thing for the ref photos.
                ref_img_output_path = os.path.join(
                    predicted_label_folder, os.path.basename(ref_img_input_path)
                )

                # print(f'Reference: {ref_img_input_path} -> {ref_img_output_path}')
                shutil.copy(ref_img_input_path, ref_img_output_path)
                result["status"] = "success"

        except Exception as e:
            result["error"] = f"Error matching embeddings: {e}"
            return result

    return result


def cluster_embeddings(
    source_cache: str,
    source_list_images: str,
    clustering_algorithm: str,
    eps: float,
    min_samples: int,
    output_path: str,
    fail_path: str,
) -> dict:
    """Cluster embeddings, and save the output. Return result status."""

    result = {}
    result['missed_count'] = 0 

    source_embeddings = [
        os.path.join(source_cache, file)
        for file in os.listdir(source_cache)
        if file.split(".")[-1] == "pkl"
    ]

    embedding_file_to_image_table = {
        file.split("/")[-1].split(".")[0]: file for file in source_list_images
    }

    loaded_embeddings = []
    loaded_faces = []
    cluster_obj = None

    # init clustering algorithm.
    if clustering_algorithm == enums.ClusteringAlgorithm.DBSCAN.value:
        cluster_obj = c_algorithm.DBSCAN(
            eps=eps, min_samples=min_samples, metric="euclidean", leaf_size=50
        )
    elif clustering_algorithm == enums.ClusteringAlgorithm.OPTICS.value:
        cluster_obj = c_algorithm.OPTICS(min_samples=min_samples, metric="euclidean")

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

        try:
            embedding_dict = read_embeddingpkl(os.path.join(source_cache, file))
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

        #print POSTPROCESS_PROGRESS
        progress =math.ceil(50 + ((i + 1) / len(labels) * 50))

        if int(progress) % 2 == 0:
            print(f"POSTPROCESS_PROGRESS: {int(progress)}")

        # find the original image from look up table.
        backtracked_file_name = face_to_embedding_file_table[i]

        source_img_path = embedding_file_to_image_table[
            backtracked_file_name.split("/")[-1].split(".")[0]
        ]

        # label -1 indicates failed clustering.
        if label == -1:
            failed_img_output_path = os.path.join(
                fail_path, os.path.basename(source_img_path)
            )
            shutil.copy(source_img_path, failed_img_output_path)
            print(
                f"Failed clustering on {source_img_path}. Saved to {failed_img_output_path}"
            )
            result['missed_count'] += 1
            continue

        label_folder = os.path.join(output_path, "subject-no-" + str(label))
        os.makedirs(label_folder, exist_ok=True)

        source_img_output_path = os.path.join(
            label_folder, os.path.basename(source_img_path)
        )

        shutil.copy(source_img_path, source_img_output_path)

        # save face sample for sanity check.
        try:            
            aligned_face_sample_output_path = os.path.join(
                label_folder, "target_face.jpg"
            )
            cv2.imwrite(aligned_face_sample_output_path, loaded_faces[i])
        except Exception as e:
            result["error"] = f"Error saving aligned face sample: {e}"
            return result

    result["status"] = "success"

    return result


if __name__ == "__main__":
    pass