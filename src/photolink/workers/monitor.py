import threading
import time
import sys
import traceback
from photolink.workers import WorkerSignals
import multiprocessing as mp
import photolink.utils.enums as enums
from photolink import get_application_path, get_config_file
from photolink.utils.function import read_config
from pathlib import Path
import json


class ProgressMonitor(threading.Thread):
    """Periodically monitors the progress by checking the number of files generated."""
    def __init__(self, task: enums.Task, stop_event: mp.Event, signals: WorkerSignals, monitor_interval: int):
        super().__init__()
        self.monitor_interval = monitor_interval
        self.stop_event = stop_event
        self.signals = signals
        self.task = task
        self.progress = 0

        # call some basic path info.
        self.application_path = get_application_path()
        config = get_config_file()
        self.config = read_config(config)
        self.cache_dir = self.application_path / ".cache"
        self.source_cache = self.cache_dir / "source"
        self.reference_cache = self.cache_dir / "reference"
        self.job_path = self.cache_dir / Path("job.json")
        self.job = self.read_json_file(self.job_path)
        self.output_path = Path(self.job["output"])
        self.log_file = self.application_path / "worker.log"

    def run(self):
        """Run the progress monitor."""
        with open(self.log_file, "r") as f:
            while not self.stop_event.is_set():
                try:
                    where = f.tell()
                    line = f.readline()
                    if not line:
                        f.seek(where)
                    else:
                        # retrieve the name of the image that just got processed.
                        progress_input = ' '.join(line.split(':')[-2:])
                        self.signals.result.emit(progress_input)

                except Exception as e:
                    exctype, value, tb = sys.exc_info()
                    self.signals.error.emit((exctype, value, traceback.format_exc()))
                time.sleep(self.monitor_interval)

    def send_update_signals(self, value):
        """Send the update signals only when it makes sense."""
        value = int(value)
        try:
            if value >= self.progress:
                self.progress = value
                self.signals.progress.emit(self.progress)
        except Exception as e:
            exctype, value, tb = sys.exc_info()
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        
        # keep a bit of pause.
        time.sleep(self.monitor_interval)

    def read_json_file(self, file_path):
        with open(str(file_path), 'r') as file:
            data = json.load(file)
        return data