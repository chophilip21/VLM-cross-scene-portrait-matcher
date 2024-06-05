import photomatcher.model.yunet as yunet
import photomatcher.model.sface as sface
import os
import shutil
from functools import partial
import multiprocessing as mp
import pickle

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
    save_embedding_name = os.path.join(save_path, os.path.basename(image_path).split(".")[0] + ".pkl")

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
        