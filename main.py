from photolink.pipeline.main import main
import os
import sys
from pathlib import Path


def set_env_variables():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.join(base_dir, 'env')

    # Set environment variables to use the virtual environment
    os.environ['PYTHONHOME'] = venv_dir
    os.environ['PYTHONPATH'] = os.path.join(venv_dir, 'Lib', 'site-packages')

    # Add virtual environment's Scripts directory to PATH
    os.environ['PATH'] = os.path.join(venv_dir, 'Scripts') + os.pathsep + os.environ['PATH']



if __name__ == "__main__":
    main()