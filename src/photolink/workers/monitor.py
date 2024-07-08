"""Thread just for monitoring the worker progress of worker thread. This is to prevent the GUI from freezing."""

import threading
import time
import os
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
    def __init__(self, task: enums.Task, stop_event: mp.Event, signals: WorkerSignals, monitor_interval: int, oputput_stream):
        super().__init__()
        self.monitor_interval = monitor_interval
        self.stop_event = stop_event
        self.signals = signals
        self.task = task
        self.progress = 0
        self.output_stream = oputput_stream

        # call some basic path info.
        self.application_path = get_application_path()
        config = get_config_file(self.application_path)
        self.config = read_config(config)
        self.cache_dir = self.application_path / ".cache"
        self.source_cache = self.cache_dir / "source"
        self.reference_cache = self.cache_dir / "reference"
        self.job_path = self.cache_dir / Path("job.json")
        self.job = self.read_json_file(self.job_path)
        self.outout_path = Path(self.job["output"])

        if self.task == enums.Task.SAMPLE_MATCHING.name:
            self.preprocess_total = len(self.job["source"]) + len(self.job["reference"])

            self.num_subjects = len(self.job["source"])

        elif self.task == enums.Task.CLUSTERING.name:
            self.preprocess_total = len(self.job["source"])
        else:
            raise ValueError(f'Progress monitor: {self.task} is not a valid choice')

        self.output_stream = OutputStream()
        self.output_stream.write('BITCH!!!!!!!!!!!!!!!!.\n')
        print('Progress monitor initialized.', flush=True)

    def run(self):
        while not self.stop_event.is_set():
            try:
                # Capture stdout
                with self.output_stream.lock:
                    output = self.output_stream.getvalue()
                    self.output_stream.truncate(0)
                    self.output_stream.seek(0)

                if output:
                    self.signals.progress.emit(output.strip())  # Emit stripped output to remove extra newlines

            except Exception as e:
                exctype, value, tb = sys.exc_info()
                self.signals.error.emit((exctype, value, traceback.format_exc()))
            time.sleep(self.monitor_interval)

            # if self.task == enums.Task.SAMPLE_MATCHING.name:
                
            #     if not preprecess_done:

            #         num_preprocessed = len(os.listdir(str(self.source_cache))) + len(os.listdir(str(self.reference_cache)))

            #         progress = int((num_preprocessed / self.preprocess_total) * 50)

            #         self.send_update_signals(progress)

            #         # stop preprocessing monitor
            #         if num_preprocessed >= self.preprocess_total:
            #             preprecess_done = True

            #     else:
            #         # postprocess updates. first check how many subject folder is generated.

            #         for folder in os.listdir(str(self.outout_path)):
            #             if folder.startswith(enums.OutputPrefix.MATCH.name):
            #                 self.num_subjects += 1



            # elif self.task == enums.Task.CLUSTERING.name:
                
            #     if not preprecess_done:
            #         num_preprocessed = len(os.listdir(str(self.source_cache)))

            #         progress = int((num_preprocessed / self.preprocess_total) * 50)

            #         self.send_update_signals(progress)

            #         # stop preprocessing monitor
            #         if num_preprocessed >= self.preprocess_total:
            #             preprecess_done = True
                
            #     else:
            #         # post process update
            #         pass


            # else:
            #     raise NotImplementedError(f'Progress monitor: {self.task} is not a valid choice')



    def send_update_signals(self, value):
        """Send the update signals only when it makes sense."""
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