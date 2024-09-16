"""Modules to communicate with Dropbox API."""

import os
import threading
from pathlib import Path

import dropbox
from loguru import logger


class DropboxClient:
    """
    Thread-safe, lazy-loaded singleton Dropbox client.

    This class provides a singleton Dropbox client instance, initialized lazily upon first access.
    """

    _client = None
    _lock = threading.Lock()

    def __init__(self):
        """Initialize variables."""
        pass  # No need to initialize _client here since it's a class variable

    @property
    def client(self):
        """Lazily initialize the Dropbox client."""
        if self._client is None:
            with self._lock:
                if self._client is None:
                    # Retrieve the access token securely
                    access_token = os.getenv("DROPBOX_ACCESS_TOKEN")
                    if not access_token:
                        raise ValueError(
                            "Dropbox access token must be set in the 'DROPBOX_ACCESS_TOKEN' environment variable."
                        )

                    # Initialize the Dropbox client
                    self._client = dropbox.Dropbox(access_token)
                    logger.info("Dropbox client initialized.")

        return self._client


client = DropboxClient().client


def download_file(dropbox_path: str, local_path: Path):
    """
    Download a file from Dropbox to a local path.

    Args:
        dropbox_path (str): The path to the file in Dropbox.
        local_path (Path): The local path where the file should be saved.

    Raises:
        dropbox.exceptions.ApiError: If an error occurs during the download.
    """
    try:
        logger.info(f"Downloading '{dropbox_path}' from Dropbox to '{local_path}'.")
        metadata, res = client.files_download(path=dropbox_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(res.content)
        logger.info(f"Downloaded '{dropbox_path}' to '{local_path}'.")
    except dropbox.exceptions.ApiError as err:
        logger.error(f"Failed to download '{dropbox_path}' from Dropbox: {err}")
        raise


if __name__ == "__main__":
    print("testing dropbox...")

    download_file(
        "/test.jpg",
        Path("test.jpg"),
    )
