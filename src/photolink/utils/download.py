"""All modules related to downloading weights would go here."""

from photolink import get_application_path, get_config
from huggingface_hub import hf_hub_download, list_repo_files
import os
from loguru import logger
import IPython


class Local:

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Local, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self._file_list = None
        self._repo_id = None
        self._repo_token = None

    @property
    def file_list(self):
        """Lazy load the file list."""
        if self._file_list is None:
            config = get_config()
            repo_id = str(config.get("HUGGINGFACE", "REPO_ID"))
            repo_token = str(config.get("HUGGINGFACE", "REPO_TOKEN"))

            files_into_repo = list_repo_files(repo_id, use_auth_token=repo_token)

            self._file_list = files_into_repo
            self._repo_id = repo_id
            self._repo_token = repo_token

        return self._file_list

    def get_token(self):
        """Get the token from the config."""
        return self._repo_token

    def get_repo_id(self):
        """Get the repo id from the config."""
        return self._repo_id


local = Local()  # Singleton instance of Local


def check_weights_exist(local_path, remote_path, is_folder=False):
    """Check if weights exist locally, if not download from remote path. Ensure path compatibility b/w linux and windows."""
    application_path = get_application_path()
    local_path = os.path.join(application_path, str(local_path))

    if not os.path.exists(local_path) or len(os.listdir(local_path)) == 0:
        logger.info(
            f"Weights for {str(local_path)} not found. Downloading from {str(remote_path)}"
        )

        try:

            if is_folder:
                os.makedirs(local_path, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(local_path), exist_ok=True)

            files_in_repo = local.file_list
            folder_to_download = str(remote_path)
            local_dir = os.path.dirname(local_path)
            files_to_download = [
                file for file in files_in_repo if folder_to_download in file
            ]

            for file_name in files_to_download:

                hf_hub_download(
                    repo_id=local.get_repo_id(),
                    filename=file_name,
                    use_auth_token=local.get_token(),
                    local_dir=local_dir,
                    force_download=True,
                )

            logger.info(
                f"Weights downloaded successfully for model : {str(local_path)}"
            )

        except Exception as e:
            logger.error(f"Error downloading weights for {str(local_path)} model : {e}")
            return
    else:
        logger.info(f"Weights found locally for {str(local_path)}")
        return
