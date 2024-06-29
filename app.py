from photolink.pipeline.main import run
from pathlib import Path

appication_path = None

def get_application_path():
    """Singleton pattern for application path."""
    global appication_path
    if appication_path is None:
        appication_path = Path(__file__).resolve().parent
    return appication_path

def get_config_file(application_path: Path) -> Path:
    """Get the config file path."""
    return application_path / Path("config.ini")


if __name__ == "__main__":
    run()