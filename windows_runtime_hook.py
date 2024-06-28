"""Dirty fix to prevent Windows using wrong Python.exe on subprocess."""
import os
from pathlib import Path
from photolink.pipeline.main import get_application_path, get_config_file
from photolink.utils.function import read_config

# Adjust these paths based on your virtual environment
app_path = get_application_path()
config = read_config(get_config_file(app_path))
venv_path = app_path / Path(config["WINDOWS"]["VIRTUAL_ENV"])
os.environ['PYTHONHOME'] = str(venv_path.relative_to(app_path))
os.environ['PYTHONPATH'] = str(venv_path  / 'Lib' / 'site-packages')
os.environ['PATH'] = str(venv_path / 'Scripts') + os.pathsep + os.environ['PATH']
