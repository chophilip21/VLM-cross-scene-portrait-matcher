import configparser
import glob
import hashlib
import lzma
import os
import pickle

from loguru import logger
from pathlib import Path

import photolink.utils.enums as enums
import shutil
from datetime import datetime
import json


def search_all_images(path: Path):
    """Recursively search all images in a directory. """
    images = []
    path_ = path + '/**/*.*'
    files = glob.glob(path_,recursive = True) 

    for file in files:
    
        if file.split('.')[-1].lower() in enums.IMAGE_EXTENSION:
            images.append(file)

    return images

def search_all_xz_file(path: Path)-> list:
    """Recursively search all embeddings file in a directory. """
    embeddings = []
    path_ = str(path) + '/**/*.*'
    files = glob.glob(path_,recursive = True) 

    for file in files:
        if file.split('.')[-1].lower() == 'xz':
            embeddings.append(file)

    return embeddings


def read_config(file)-> dict:
    """Read config file"""
    config = configparser.ConfigParser()
    config.read(file)
    return config

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
    """Create hash based on path, or bytes. Avoid multiple opening by using bytes input"""

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
    for item in directory.iterdir():
        # we only care about directories. Not files.
        if item.is_dir():
            shutil.rmtree(item)

def get_current_date()->str:
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
        with open(file_path, 'r') as json_file:
            return json.load(json_file)
    else:
        # create empty file if it does not exist
        with open(file_path, 'w') as json_file:
            json.dump({}, json_file)


def get_hash_image_pair_list(data: list)-> list:
    """Return a list of dict, hash to image path pair."""

    converted_data = []

    for image_path in data:
        hash_ = checksum(image_path)
        tmp = {}
        tmp[hash_] = image_path
        converted_data.append(tmp)

    return converted_data




    