import configparser
import os
import sys
from pathlib import Path


class SingletonPath:
    """Modules to get the application path and config file path."""

    def __init__(self):
        self._application_path = None
        self._config_file_path = None
        self._config = None

    @property
    def application_path(self):
        """Get the application path."""

        if self._application_path is None:

            if "NUITKA_ONEFILE_EXE" in os.environ or getattr(sys, "frozen", False):
                self._application_path = Path(sys.executable).parent

            else:

                self._application_path = Path(__file__).resolve().parents[2]

        return self._application_path

    @property
    def config_file(self) -> Path:
        """Get the config file path."""

        if self._config_file_path is None:

            self._config_file_path = Path(
                self.application_path / "assets" / "config.ini"
            )

            if not self._config_file_path.exists():
                raise FileNotFoundError(
                    f"Config file {self._config_file_path} not found. Exiting..."
                )

        return self._config_file_path

    @property
    def config(self):
        """Read the config file."""
        if self._config is None:
            self._config = read_config(self.config_file)
        return self._config


configuration = SingletonPath()


# should be used when using pyinstaller
def get_application_path() -> Path:
    """Get the application path."""
    return configuration.application_path


def read_config(file) -> dict:
    """Read config file"""
    config = configparser.ConfigParser()
    config.read(file)
    return config


def get_config() -> dict:
    """Get the config file from cache."""
    return configuration.config
