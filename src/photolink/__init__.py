from pathlib import Path
import sys

class SingletonPath:
    """Modules to get the application path and config file path."""

    def __init__(self):
        self._application_path = None
        self._config_file_path = None

    @property
    def application_path(self):
        """Get the application path."""

        if self._application_path is None:
            
                # Depending on how we launch application, the root changes somehow. Dirty fix to prevent that. 
                # self._application_path = Path(__file__).resolve().parent.parents[2]

             
                # if (self._application_path / "photolink.exe").exists():
                #     return self._application_path

                # print(f"Application path: {self._application_path / 'photolink.exe'} could not be found. You are running locally")

                # # If photolink.exe does not exist, we must be running locally.
            self._application_path = Path(__file__).resolve().parents[2]
            
                # # check pyproject.toml. If not exists again, something is worng
                # if not (self._application_path / "pyproject.toml").exists():
                #     raise FileNotFoundError(f"Application path '{self._application_path}' is not set properly. Exiting...")
            
                # print(f"Application path: {self._application_path} found. Running locally")

        return self._application_path

    @property
    def config_file(self) -> Path:
        """Get the config file path."""

        if self._config_file_path is None:

            self._config_file_path = Path(self._application_path / 'assets' / "config.ini")

            if not self._config_file_path.exists():
                raise FileNotFoundError(f"Config file {self._config_file_path} not found. Exiting...")

        return self._config_file_path



path = SingletonPath()

# should be used when using pyinstaller
def get_application_path()->Path:
    """Get the application path."""
    return path.application_path

def get_config_file() -> Path:
    """Get the config file path."""
    return path.config_file