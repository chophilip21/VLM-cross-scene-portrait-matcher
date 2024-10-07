"""Module to monitor the progress of the worker."""

import json
import sys
import threading
import time
import traceback
from pathlib import Path

import gdown
from loguru import logger

import photolink.utils.enums as enums
from photolink import get_application_path, get_config
from photolink.workers import WorkerSignals


class ProgressMonitor(threading.Thread):
    """Periodically monitors the progress by checking the number of files generated.

    Also includes the mechanism to download the weights if they are not already downloaded.
    """

    def __init__(
        self,
        task: enums.Task,
        stop_event: threading.Event,
        signals: WorkerSignals,
        monitor_interval: int,
    ):
        super().__init__()
        self.monitor_interval = monitor_interval
        self.stop_event = stop_event
        self.signals = signals
        self.task = task
        self.progress = 0

        # Call some basic path info.
        self.application_path = get_application_path()
        self.config = get_config()
        self.cache_dir = self.application_path / ".cache"
        self.source_cache = self.cache_dir / "source"
        self.reference_cache = self.cache_dir / "reference"
        self.job_path = self.cache_dir / Path("job.json")
        self.job = self.read_json_file(self.job_path)
        self.output_path = Path(self.job["output"])
        self.log_file = self.application_path / "worker.log"

    def run(self):
        """Run the progress monitor."""

        # check if there is any weights to be downloaded
        if self.task == enums.Task.DP2_MATCH.name:
            local_path = str(
                self.application_path
                / Path(self.config.get("YOLOSEG", "LOCAL_PATH"))
            )

            remote_path = str(self.config.get("YOLOSEG", "REMOTE_PATH"))

            if not Path(local_path).exists():
                logger.info("Downloading weights for yolo seg model.")
                self.storage_download(remote_path, local_path, "YOLOSEG")

        with open(self.log_file, "r") as f:
            while not self.stop_event.is_set():
                try:
                    where = f.tell()
                    line = f.readline()
                    if not line:
                        f.seek(where)
                    else:
                        # Retrieve the name of the image that just got processed.
                        progress_input = " ".join(line.split(":")[-2:])

                        # TODO: If you want, you can send info about progress here
                        # self.signals.result.emit(progress_input)

                except Exception as e:
                    exctype, value, tb = sys.exc_info()
                    self.signals.error.emit((exctype, value, traceback.format_exc()))
                time.sleep(self.monitor_interval)

    def read_json_file(self, file_path):
        with open(str(file_path), "r") as file:
            data = json.load(file)
        return data

    def storage_download(self, remote_path: str, local_path: str, model_name: str):
        """Download the file from the remote storage to the local storage.

        Args:
            remote_path (str): Path to the file in the remote storage.
            local_path (str): Path to the file in the local storage.
            model_name (str): Name of the model for which the weights are being downloaded.
        """
        try:
            gdown.download(remote_path, local_path, quiet=False, fuzzy=True)
            logger.info(f"Weights downloaded successfully for model : {model_name}")
            self.signals.result.emit(f"{model_name} weights downloaded successfully.")

        except Exception as e:
            logger.error(f"Error downloading weights for {model_name} model : {e}")
            exctype, value, tb = sys.exc_info()
            self.signals.error.emit((exctype, value, traceback.format_exc()))
            return
