"""All modules related to downloading weights would go here."""

import os

from huggingface_hub import hf_hub_download, list_repo_files
from loguru import logger

from photolink import get_application_path, get_config


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


def check_weights_exist(local_path, remote_path):
    """Check if weights exist locally, if not download from remote path. Ensure path compatibility b/w linux and windows."""
    application_path = get_application_path()
    local_path = os.path.join(application_path, str(local_path))
    save_loc = None

    if not os.path.exists(local_path):
        logger.info(
            f"Weights for {str(local_path)} not found. Downloading from {str(remote_path)}"
        )

        files_in_repo = local.file_list
        folder_to_download = str(remote_path)
        local_base = os.path.basename(local_path)

        # treat these differentlty.
        if ".mlpackage" in local_base or len(local_base.split(".")) == 1:
            save_loc = os.path.dirname(local_path)

        # most cases it will be just single file.
        elif len(os.path.basename(local_path).split(".")) == 2:
            save_loc = os.path.join(os.path.dirname(local_path), "../")

        else:
            logger.error(f"Invalid local path : {str(local_path)}")
            raise ValueError(f"Invalid local path : {str(local_path)}")

        # Check again.
        if save_loc is None:
            logger.error(f"Invalid local path : {str(local_path)}")
            raise ValueError(f"Invalid local path : {str(local_path)}")

        try:
            files_to_download = [
                file for file in files_in_repo if folder_to_download in file
            ]

            if len(files_to_download) == 0:
                raise ValueError(
                    "No files found in the remote path, cannot be possible!"
                )

            for file_name in files_to_download:

                hf_hub_download(
                    repo_id=local.get_repo_id(),
                    filename=file_name,
                    use_auth_token=local.get_token(),
                    local_dir=save_loc,
                    force_download=True,
                )

            logger.info(
                f"Weights downloaded successfully for model : {str(local_path)}"
            )

            return True
        except Exception as e:
            # TODO: Raise proper error to the client.
            logger.error(
                f"Error downloading weights for {str(local_path)} model : {e}. Removing because failed download."
            )

            # delete the entire folder.
            import shutil

            shutil.rmtree(local_path)
            return False
    else:
        logger.info(f"Weights found locally for {str(local_path)}")
        return True
