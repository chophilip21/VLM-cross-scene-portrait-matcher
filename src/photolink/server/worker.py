"""ML algorithms for face detection and recognition, and clustering, (multiprocessing) worker functions."""
import photolink.utils.enums as enums
import os
import shutil
import pickle
import numpy as np
import faiss
import sklearn.cluster as c_algorithm
import hdbscan
import cv2
from pathlib import Path
import math
from PySide6.QtCore import QRunnable



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

        data["faces"] = np.array(embedding_dict["faces"])
        data["embeddings"] = embeddings

    return data


def match_embeddings(
    source_cache: Path,
    reference_cache: Path,
    source_list_images: list,
    reference_list_images: list,
    output_path: Path,
) -> dict:
    """Match the embeddings and save the match. Return result status."""

    result = {}

    faiss_index = faiss.IndexFlatL2(128)
    source_embeddings = [
        source_cache / Path(file)
        for file in os.listdir(source_cache)
        if file.split(".")[-1] == "pkl"
    ]

    reference_embeddings = [
        reference_cache / Path(file)
        for file in os.listdir(reference_cache)
        if file.split(".")[-1] == "pkl"
    ]

    # create look up table using Pathlib.
    source_dict = {Path(file).stem: file for file in source_list_images}
    reference_dict = {Path(file).stem: file for file in reference_list_images}

    # start by reading each embedding file, and adding to faiss index.
    for i, embedding_file in enumerate(source_embeddings):

        progress = math.ceil(50 + ((i + 1) / len(source_embeddings) * 25))
        print(f"Post-Processing:{int(progress)}", flush=True)

        try:
            source_embedding_dict = read_embeddingpkl(source_cache / Path(embedding_file))

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
        print(f"Post-Processing:{int(progress)}", flush=True)

        try:
            reference_embedding_dict = read_embeddingpkl(reference_cache / Path(file))

            reference_embedding = reference_embedding_dict["embeddings"]

            for face_embedding in reference_embedding:

                # check number of dimensions. Should be (1, 128)
                if face_embedding.ndim == 1:
                    face_embedding = np.expand_dims(face_embedding, axis=0)

                # search for the nearest embedding in the source embeddings.
                D, I = faiss_index.search(face_embedding, 1)
                distance = D[0][0]

                # predicted label must exist in the lookup table.
                predicted_label = Path(source_embeddings[I[0][0]]).stem

                if predicted_label not in source_dict:
                    raise ValueError(f"Predicted label {predicted_label} not found in source_dict.")

                predicted_source_input_path = source_dict[predicted_label]

                # now get the equivalent reference image.
                reference_label = Path(file).stem

                if reference_label not in reference_dict:
                    raise ValueError(
                        f"Reference label {reference_label} not found in reference_dict."
                    )

                ref_img_input_path = reference_dict[reference_label]

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
    source_list_images: list,
    clustering_algorithm: str,
    eps: float,
    min_samples: int,
    output_path: Path,
    fail_path: Path,
) -> dict:
    """Cluster embeddings, and save the output. Return result status."""

    result = {}
    result["missed_count"] = 0

    source_embeddings = [
        source_cache / Path(file)
        for file in os.listdir(source_cache)
        if file.split(".")[-1] == "pkl"
    ]

    embedding_file_to_image_table = {
        Path(file).stem: file for file in source_list_images
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
        raise NotImplementedError(f"Clustering algorithm {clustering_algorithm} not supported.")

    # because there are multiple faces, we need to keep track of the embeddings.
    face_to_embedding_file_table = {}
    index_count = 0
    for file in source_embeddings:

        try:
            embedding_dict = read_embeddingpkl(source_cache / Path(file))
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

        # print POSTPROCESS_PROGRESS
        progress = math.ceil(50 + ((i + 1) / len(labels) * 50))
        print(f"Post-Processing:{int(progress)}", flush=True)

        # find the original image from look up table.
        backtracked_file_name = Path(face_to_embedding_file_table[i])

        source_img_path = embedding_file_to_image_table[
            backtracked_file_name.stem
        ]

        # label -1 indicates failed clustering.
        if label == -1:
            failed_img_output_path = str(fail_path / Path(os.path.basename(source_img_path)))
            shutil.copy(source_img_path, failed_img_output_path)
            print(f"Failed clustering on {source_img_path}. Saved to {failed_img_output_path}")
            result["missed_count"] += 1
            continue

        label_folder = output_path / Path("subject-no-" + str(label))
        label_folder.mkdir(parents=True, exist_ok=True)

        source_img_output_path = str(label_folder / Path(os.path.basename(source_img_path)))

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


if __name__ == "__main__":
    pass
