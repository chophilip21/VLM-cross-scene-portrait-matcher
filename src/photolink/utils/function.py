import photolink.utils.enums as enums
import glob
import configparser
import os


def search_all_images(path):
    """Recursively search all images in a directory. Select one if choose_one is True."""
    images = []
    path_ = path + '/**/*.*'
    files = glob.glob(path_,recursive = True) 
    for file in files:
        
        if file.split('.')[-1].lower() in enums.IMAGE_EXTENSION:
            images.append(file)

    return images

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

