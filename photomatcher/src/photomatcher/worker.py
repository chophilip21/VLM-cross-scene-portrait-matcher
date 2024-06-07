"""ML algorithms for face detection and recognition, and clustering."""
import photomatcher.model.yunet as yunet
import photomatcher.model.sface as sface
import photomatcher.enums as enums
import os
import shutil
from functools import partial
import multiprocessing as mp
import pickle
import numpy as np
import faiss
import sklearn.cluster as c_algorithm
import hdbscan


def _run_ml_model(image_path: str, save_path: str, fail_path: str):
    """Process face detection and recognition for images, save embeddings to save_path, otherwide save the entire image to fail_path."""
    detection_result = yunet.run_face_detection(image_path)

    failed_image = os.path.join(fail_path, os.path.basename(image_path))

    if "error" in detection_result:
        shutil.copy(image_path, failed_image)
        print(
            f"Face detection error on source image {image_path}: {detection_result['error']}"
        )

    image = detection_result["image"]
    embedding_dict = sface.run_face_recognition(image, detection_result["faces"])

    if "error" in embedding_dict:
        shutil.copy(image_path, failed_image)
        print(
            f"Face recognition error on source image {image_path}: {embedding_dict['error']}"
        )

    embedding = embedding_dict["embeddings"]
    save_embedding_name = os.path.join(
        save_path, os.path.basename(image_path).split(".")[0] + ".pkl"
    )

    # pickle dump embedding.
    with open(save_embedding_name, "wb") as f:
        pickle.dump(embedding, f)

    return True


def run_model_mp(
    entries, num_workers: int, chunksize: int, save_path: str, fail_path: str
):
    """Use multiprocessing to run the model."""
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=num_workers)

    with pool as p:
        result = list(
            p.imap_unordered(
                partial(_run_ml_model, save_path=save_path, fail_path=fail_path),
                entries,
                chunksize=chunksize,
            )
        )


def read_embedding(embedding_path) -> np.ndarray:
    """Read and validate embeddings from pickle path."""
    with open(embedding_path, "rb") as f:
        embedding = pickle.load(f)

        # basic validation
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        else:
            raise ValueError(
                f"Error on {embedding_path}. Embedding must be a list, not {type(embedding)}"
            )

        if embedding.shape != (1, 128):
            raise ValueError(
                f"Error on {embedding_path}. Embedding must be a (1, 128) numpy array, not {embedding.shape}"
            )

        return embedding


def match_embeddings(
    source_cache: str,
    reference_cache: str,
    source_list_images: list,
    reference_list_images: list,
    output_path: str,
)-> dict:
    """Match the embeddings and save the match. Return result status."""

    result = {}

    try:
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

        for file in source_embeddings:
            embedding = read_embedding(os.path.join(source_cache, file))
            faiss_index.add(embedding)

        print("added all the source embeddings to faiss index...")

        for file in reference_embeddings:
            embedding = read_embedding(os.path.join(reference_cache, file))

            D, I = faiss_index.search(embedding, 1)
            distance = D[0][0]

            # predicted label must exist in the lookup table.
            predicted_label = source_embeddings[I[0][0]].split("/")[-1].split(".")[0]

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
                predicted_label_folder, os.path.basename(predicted_source_input_path)
            )

            shutil.copy(predicted_source_input_path, predicted_source_output_path)

            # same thing for the ref photos.
            ref_img_output_path = os.path.join(
                predicted_label_folder, os.path.basename(ref_img_input_path)
            )

            shutil.copy(ref_img_input_path, ref_img_output_path)
            result['status'] = "success"
        
    except Exception as e:
        return {"error": 'matching failed because of ' + str(e)}

    return result


def cluster_embeddings(
    source_cache: str,
    source_list_images: str,
    clustering_algorithm: str,
    eps: float,
    min_samples: int,
    output_path: str,
    fail_path: str,
)-> dict:
    """Cluster embeddings, and save the output. Return result status."""

    result = {}

    try:
        source_embeddings = [
            os.path.join(source_cache, file)
            for file in os.listdir(source_cache)
            if file.split(".")[-1] == "pkl"
        ]

        # Create quick look up table for the source and reference images.
        source_dict = {
            file.split("/")[-1].split(".")[0]: file for file in source_list_images
        }

        loaded_embeddings = []
        for file in source_embeddings:
            embedding = read_embedding(os.path.join(source_cache, file))
            loaded_embeddings.append(embedding.flatten())

        embeddings = np.array(loaded_embeddings)
        print(f"Embeddings shape for clustering: {embeddings.shape}")

        if clustering_algorithm == enums.ClusteringAlgorithm.DBSCAN.value:
            db = c_algorithm.DBSCAN(
                eps=eps, min_samples=min_samples, metric="euclidean", leaf_size=50
            )

            db.fit_predict(embeddings)
            labels = db.labels_

        elif clustering_algorithm == enums.ClusteringAlgorithm.OPTICS.value:
            optics = c_algorithm.OPTICS(min_samples=min_samples, metric="euclidean")
            labels = optics.fit_predict(embeddings)

        elif clustering_algorithm == enums.ClusteringAlgorithm.HDBSCAN.value:
            clustering = hdbscan.HDBSCAN(min_samples=min_samples)
            labels = clustering.fit_predict(embeddings)
        else:
            raise NotImplementedError(
                f"Clustering algorithm {clustering_algorithm} not supported."
            )
        
        # now save the output.
        for i, label in enumerate(labels):
            
            # find the original image from look up table.
            source_img_path = source_dict[source_embeddings[i].split("/")[-1].split(".")[0]]

            # label -1 indicates failed clustering.
            if label == -1:
                failed_img_output_path = os.path.join(fail_path, os.path.basename(source_img_path))
                shutil.copy(source_img_path, failed_img_output_path)
                print(f"Failed clustering on {source_img_path}. Saved to {failed_img_output_path}")
                continue

            label_folder = os.path.join(output_path, str(label))
            os.makedirs(label_folder, exist_ok=True)
           
            source_img_output_path = os.path.join(
                label_folder, os.path.basename(source_img_path)
            )

            shutil.copy(source_img_path, source_img_output_path)
            print(f"Saved {source_img_path} to {source_img_output_path}")

        result['status'] = "success"

    except Exception as e:
        return {"error": 'clustering failed because of ' + str(e)}

    return result


if __name__ == "__main__":
    pass
