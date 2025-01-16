from pathlib import Path

import cv2
import numpy as np
import onnxruntime
from loguru import logger
from PIL import Image

from photolink import get_application_path, get_config
from photolink.models.geometry import SimilarityTransform
from photolink.utils.download import check_weights_exist

# Reference 5-point target landmarks for alignment (e.g., 112x112 ArcFace style)
LANDMARKS_TARGET = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


class Local:
    """
    Singleton for FaceTransformer (Octuplet Loss) ONNX model.
    Lazily loads the model so we don't start an ONNX session prematurely.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Local, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self._session = None
        self._input_name = None
        self._output_names = None

    @property
    def session(self):
        """
        Lazy initialization of the FaceTransformer ONNX model.
        """
        if self._session is None:
            application_path = get_application_path()
            config = get_config()

            # Read model path from config
            model_path = application_path / Path(
                config["FACETRANSFORMER"]["LOCAL_PATH"]
            )
            remote_path = str(config["FACETRANSFORMER"]["REMOTE_PATH"])

            downloaded = check_weights_exist(model_path, remote_path)
            if not downloaded:
                raise MemoryError("Could not download the model for face recognition")

            # Create session
            providers = ["CPUExecutionProvider"]  # or GPUExecutionProvider if available
            logger.info(f"Loading FaceTransformer model from: {model_path}")
            self._session = onnxruntime.InferenceSession(
                str(model_path), providers=providers
            )
            # Save input / output names for quick reference
            self._input_name = self._session.get_inputs()[0].name
            self._output_names = [out.name for out in self._session.get_outputs()]

        return self._session

    @property
    def input_name(self):
        if self._input_name is None:
            _ = self.session  # Force session creation
        return self._input_name

    @property
    def output_names(self):
        if self._output_names is None:
            _ = self.session  # Force session creation
        return self._output_names


def _align_face(src_img: np.ndarray, src_landmarks: np.ndarray) -> np.ndarray:
    """
    Align the face to a canonical 112x112 using 5 input landmarks
    and LANDMARKS_TARGET references.
    """
    # Estimate similarity transform from the user-provided landmarks to the target
    tform = SimilarityTransform()
    tform.estimate(src_landmarks, LANDMARKS_TARGET)
    # Warp
    matrix = tform.params[0:2, :]
    aligned = cv2.warpAffine(
        src_img, matrix, (112, 112), borderValue=0.0, flags=cv2.INTER_LINEAR
    )
    return aligned


def run_face_recognition(
    cropped_img: np.ndarray,
    five_landmarks: np.ndarray,
) -> np.ndarray:
    """
    Given:
      - cropped_img (BGR image containing a face of arbitrary size)
      - five_landmarks (np.ndarray of shape [5,2] in the same coord space as cropped_img)

    1) Align the face to 112x112 with standard 5-point alignment
    2) Preprocess and run ONNX inference to get an embedding

    Returns:
      - embedding (np.ndarray): The face embedding vector
    """

    if isinstance(cropped_img, Image.Image):
        cropped_img = np.array(cropped_img)

    # 1. Align the face using the 5 landmarks
    aligned_face = _align_face(cropped_img, five_landmarks)

    # 2. Prepare input blob for the network
    #    - e.g., BGR -> [0..255] range, float32, shape (1,3,112,112)
    input_blob = np.asarray([aligned_face], dtype=np.float32)  # (1,112,112,3)
    # Transpose to NCHW
    input_blob = input_blob.transpose((0, 3, 1, 2))  # => (1,3,112,112)

    # 3. Run inference
    session = local.session
    input_name = local.input_name
    output_names = local.output_names

    result = session.run(output_names, {input_name: input_blob})
    # Assume the embedding is the first (or only) output
    embedding = result[0][0]

    return embedding


# Instantiate the singleton (optional)
local = Local()

if __name__ == "__main__":
    logger.info("Running sample FaceTransformer inference...")
    import copy
    import os

    from IPython import embed

    from photolink.models.facemesh import run_facemesh_inference
    from photolink.utils.function import search_all_images
    from photolink.utils.image_loader import ImageLoader

    # Collect all face images
    all_faces = search_all_images(r"C:\Users\choph\photomatcher\dataset\face")
    result = {}

    for i, face in enumerate(all_faces):
        face_img = ImageLoader(face).get_downsampled_image()
        w, h = face_img.size
        face_box = [0, 0, w, h]
        mesh_output = run_facemesh_inference(face_img, face_box)
        kpts_5 = mesh_output.get("five_keypoints_2d", None)
        if kpts_5 is None:
            logger.warning(f"No landmarks found for {face}. Skipping.")
            continue

        # run face recognition to get embedding
        embedding = run_face_recognition(face_img, kpts_5)
        result[face] = embedding

        # sanity check: visualize 5 landmarks
        figure = np.array(copy.deepcopy(face_img))
        for px, py in kpts_5:
            cv2.circle(figure, (int(px), int(py)), 3, (0, 0, 255), -1)

        figure = cv2.cvtColor(figure, cv2.COLOR_RGB2BGR)
        save_name = os.path.join("./", os.path.basename(face))
        print(f"Saving to {save_name}")
        cv2.imwrite(save_name, figure)

    # -------------------------------------------------------------------------
    # Compute cosine similarity between each pair of embeddings in `result`
    # -------------------------------------------------------------------------
    keys_list = list(result.keys())
    n = len(keys_list)

    cos_sim_dict = {}
    for i in range(n):
        for j in range(i + 1, n):
            k1 = keys_list[i]
            k2 = keys_list[j]
            emb1 = result[k1]
            emb2 = result[k2]
            # Compute cosine similarity
            dot = np.dot(emb1, emb2)
            norm = (np.linalg.norm(emb1) * np.linalg.norm(emb2)) + 1e-8
            cos_sim = dot / norm
            cos_sim_dict[(k1, k2)] = cos_sim

    # Print out pairwise similarities
    print("\nPairwise Cosine Similarities:")
    for (k1, k2), sim_val in cos_sim_dict.items():
        print(f"{os.path.basename(k1)} vs {os.path.basename(k2)}: {sim_val:.4f}")

    embed()
