import os
import sys
from pathlib import Path


class SingletonPath:
    """Modules to get the application path and config file path."""

    def __init__(self):
        self._application_path = None
        self._config_file_path = None

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
                self._application_path / "assets" / "config.ini"
            )

            if not self._config_file_path.exists():
                raise FileNotFoundError(
                    f"Config file {self._config_file_path} not found. Exiting..."
                )

        return self._config_file_path


path = SingletonPath()


# should be used when using pyinstaller
def get_application_path() -> Path:
    """Get the application path."""
    return path.application_path


def get_config_file() -> Path:
    """Get the config file path."""
    return path.config_file
