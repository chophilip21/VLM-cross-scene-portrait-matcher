import json
from pathlib import Path

from photolink import get_application_path, get_config_file
from photolink.utils.function import get_current_date, read_config

application_path = get_application_path()
config = get_config_file()
config = read_config(config)


def get_cache_dir():
    """Get the cache directory."""
    cache_dir = application_path / Path(".cache")
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def read_settings(settings_file: Path):
    """Read the settings from the settings.json file."""

    if not settings_file.exists():
        current_date = get_current_date()
        return {"save_period": 14, "last_cache_delete": current_date}

    with open(settings_file, "r") as f:
        settings = json.load(f)
    return settings


def save_dump_settings(settings_file: Path, settings_dict: dict):
    """Save the settings to the settings.json file."""

    with open(settings_file, "w") as f:
        json.dump(settings_dict, f)
