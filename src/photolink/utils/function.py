import configparser
import copy
import glob
import hashlib
import json
import lzma
import os
import pickle
import re
import shutil
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Union

import gdown
import numpy as np
from loguru import logger
from PIL import Image, ImageOps

import photolink.utils.enums as enums
from photolink import get_application_path


def _copy_image_meta(src: Image.Image, dest: Image.Image):
    preserve_metadata_keys = ["info", "icc_profile", "exif", "dpi", "applist", "format"]
    for key in preserve_metadata_keys:
        if hasattr(src, key):
            setattr(dest, key, copy.deepcopy(getattr(src, key)))
    return dest


def safe_load_image(image: Union[bytes, str], return_numpy=True) -> np.ndarray:
    """Load an image from bytes or a file path, and ensure the orientation is correct."""
    # make sure image is bytes or a valid file path
    if isinstance(image, str):
        with open(image, "rb") as f:
            image = f.read()
    elif not isinstance(image, bytes):
        raise TypeError(f"image must be bytes or a file path, not {type(image)}")
    pil_image = Image.open(BytesIO(image))

    # Make sure the orientation is correct
    if hasattr(pil_image, "_getexif") and pil_image._getexif() is not None:
        new_pil_image = ImageOps.exif_transpose(pil_image)
        pil_image = _copy_image_meta(pil_image, new_pil_image)

    if return_numpy:
        return np.array(pil_image)

    return pil_image


def search_all_images(path: Path):
    """Recursively search all images in a directory."""

    def extract_number(path):
        match = re.search(r"(?i)(\d+)(?=\.jpg)", str(path))  # Case-insensitive
        return int(match.group(0)) if match else 0

    if isinstance(path, str):
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    images = []
    files = path.glob("**/*.*")  # Using Path.glob for compatibility

    for file in files:

        extension = file.suffix.lower()[1:]

        if extension in enums.IMAGE_EXTENSION:
            images.append(str(file))  # Convert Path to str

    images = sorted(images, key=extract_number)

    return images


def search_all_xz_file(path: Path) -> list:
    """Recursively search all embeddings file in a directory."""
    embeddings = []
    path_ = str(path) + "/**/*.*"
    files = glob.glob(path_, recursive=True)

    for file in files:
        if file.split(".")[-1].lower() == "xz":
            embeddings.append(file)

    return embeddings


def config_to_env(config: configparser.ConfigParser, section: str):
    """Set some of the config variables as env variables."""

    if section not in config.sections():
        raise ValueError(f"Section {section} not found in the config file.")

    for section in config.sections():
        for key, value in config.items(section):
            key = key.upper()
            os.environ[key] = value

    return True


def compress_save(data: dict, file: str):
    """Compress and save the data to a file."""
    with lzma.open(file, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def decompress_load(file: str) -> dict:
    """Decompress and load the data from a file."""
    with lzma.open(file, "rb") as f:
        return pickle.load(f)


def checksum(
    filename, hash_factory=hashlib.blake2b, chunk_num_blocks=128, digest_size=32
):
    """Create hash based on path, or bytes. Avoid multiple opening by using bytes input. Calculating hash on file ensures file integrity, also it's cheaper because we calculate based on blocks."""

    success = True

    if hash_factory == hashlib.blake2b:
        h = hashlib.blake2b(digest_size=digest_size)
    else:
        h = hash_factory()

    if isinstance(filename, str):
        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_num_blocks * h.block_size), b""):
                h.update(chunk)
    elif isinstance(filename, bytes):
        h.update(filename)
    else:
        success = False
        logger.error(f"Input must be either bytes or string")

    return h.hexdigest() if success else None


def custom_rmtree(directory: Path):
    """Wipe out all files and directories."""

    # if hash.json exist, delete this.
    hash_file = directory / "hash.json"
    if hash_file.exists():
        logger.info(f"Deleting hash file: {hash_file}")
        os.remove(hash_file)

    for item in directory.iterdir():
        # we only care about directories. Not files.
        if item.is_dir():
            shutil.rmtree(item)


def get_current_date() -> str:
    """Return the current date as integer values for year, month, and day."""
    now = datetime.now()
    year = now.year
    month = now.month
    day = now.day
    time_string = f"{year}-{month}-{day}"
    return time_string


def read_json(file_path):
    """Read and return data from a JSON file."""
    if Path(file_path).exists():

        try:
            with open(file_path, "r") as json_file:
                return json.load(json_file) or {}
        except json.JSONDecodeError as e:
            logger.error(f"Error reading JSON file: {file_path}, {e}")
            return {}
    else:
        # create empty file if it does not exist
        with open(file_path, "w") as json_file:
            json.dump({}, json_file)

        return {}


def write_json(data, file_path):
    """Write data to a JSON file."""
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)


def read_hash_file(raise_missing_error=True):
    """Read and return the hash file (image hash-image path pairs)."""
    cache_dir = os.getenv("CACHE_DIR")

    if cache_dir is None:
        raise EnvironmentError("Please set the CACHE_DIR environment variable.")

    hash_file = Path(cache_dir) / "hash.json"

    # when postprocessing, it would be unacceptable for the hash file to be missing.
    if raise_missing_error and not hash_file.exists():
        raise FileNotFoundError(
            f"Hash file not found. It must exist by now: {hash_file}"
        )

    return read_json(hash_file)


def write_hash_file(hash_json_dict):
    """Write the hash file (image hash-image path pairs)."""
    cache_dir = os.getenv("CACHE_DIR")

    if cache_dir is None:
        raise EnvironmentError("Please set the CACHE_DIR environment variable.")

    hash_file = Path(cache_dir) / "hash.json"
    write_json(dict(hash_json_dict), hash_file)


def get_relevant_embeddings(embeddings_path: Path, job_key: str) -> list:
    """Get the relevant embeddings from the cache by screening with job.json key. Job key can be either 'source' or 'reference'. Output is list ov xz files."""

    cache_dir = os.getenv("CACHE_DIR")

    if cache_dir is None:
        raise EnvironmentError("Please set the CACHE_DIR environment variable.")

    # we need both hash table and job table to truly find out what matters for the job.
    job_file = Path(cache_dir) / "job.json"
    hash_file = Path(cache_dir) / "hash.json"

    if not job_file.exists():
        raise FileNotFoundError(f"Job file not found. Must exist by now: {job_file}")

    if not hash_file.exists():
        raise FileNotFoundError(f"Hash file not found. Must exist by now: {hash_file}")

    if job_key not in ["source", "reference"]:
        raise ValueError(f"Invalid job key: {job_key}")

    job = read_json(job_file)
    hash_ = read_json(hash_file)

    if job_key not in job:
        raise KeyError(f"Job key not found in the job file: {job_key}")

    # invert key and values in hash_ dict, so path is key and hash is value.
    hash_inverted = {v: k for k, v in hash_.items()}
    relevant_images = set(job[job_key])

    # relevant hashes, only if they are in the hash table.
    relevant_hashes = set(
        [hash_inverted[image] for image in relevant_images if image in hash_inverted]
    )

    # now we can screen out embeddings that really matters, instead of everything.
    relevant_embedding_list = [
        embeddings_path / Path(file)
        for file in os.listdir(embeddings_path)
        if file.split(".")[-1] == "xz" and file.split(".")[0] in relevant_hashes
    ]

    if not relevant_embedding_list:
        logger.error(f"No relevant embeddings found for job key: {job_key}")
    else:
        logger.info(
            f"{len(relevant_embedding_list)} Relevant embeddings found for job key: {job_key}"
        )

    return relevant_embedding_list


def check_weights_exist(local_path, remote_path):
    """Check if weights exist locally, if not download from remote path. Ensure path compatibility b/w linux and windows."""
    application_path = get_application_path()
    local = os.path.join(application_path, str(local_path))

    if not os.path.exists(local):
        logger.info(
            f"Weights for {str(local_path)} not found. Downloading from {str(remote_path)}"
        )

        try:
            gdown.download(str(remote_path), str(local), quiet=False, fuzzy=True)
            logger.info(f"Weights downloaded successfully for model : {str(local)}")

        except Exception as e:
            logger.error(f"Error downloading weights for {str(local)} model : {e}")
            return
    else:
        logger.info(f"Weights found locally for {str(local)}")
        return


def path_to_hash(path: Union[str, list]) -> str:
    """Convert the path to bytes and generate a SHA-256 hash"""

    if isinstance(path, str):
        path_bytes = path.encode("utf-8")

    elif isinstance(path, list):
        path_bytes = "".join(path).encode("utf-8")

    hash_object = hashlib.sha256(path_bytes)

    # Convert the hash to a hexadecimal string
    path_hash = hash_object.hexdigest()

    return path_hash
