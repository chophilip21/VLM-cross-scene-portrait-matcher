from pathlib import Path

appication_path = None

def get_application_path():
    """Singleton pattern for application path."""
    global appication_path
    if appication_path is None:
        appication_path = Path(__file__).resolve().parents[2]
    return appication_path

def get_config_file(application_path: Path) -> Path:
    """Get the config file path."""

    config_file_path = Path(application_path / 'assets' / "config.ini")

    if not config_file_path.exists():
        raise FileNotFoundError(f"Config file {config_file_path} not found. Exiting...")

    return config_file_path