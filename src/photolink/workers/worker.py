"""Top logic layer right below application. Execute the jobs for the worker via threading."""

from photolink.workers.jobs import JobProcessor
import threading
import multiprocessing as mp
from photolink.workers import WorkerSignals
import photolink.utils.enums as enums
import sys
import traceback


class Worker(threading.Thread):
    """Sits on top of jobs layer. Execute the jobs for the worker via threading to prevent GUI freeze. Calls the JobProcessor to run the jobs."""

    def __init__(self, identifier):
        super().__init__()
        self.identifier = identifier
        self.signals = WorkerSignals()

        # Use this to send signals to the worker.
        self._stop_event = mp.Event()

    def run(self):
        """Run the worker thread to execute the jobs."""
        try:
            job = JobProcessor(stop_event=self._stop_event, signals=self.signals)
            result = job.run()

            # Emit the result back to application
            if result == enums.StatusMessage.COMPLETE.name:
                self.signals.finished.emit()
            elif result == enums.StatusMessage.STOPPED.name:
                self.signals.stopped.emit()
            elif result == enums.StatusMessage.ERROR.name:
                self.signals.error.emit((exctype, value, traceback.format_exc()))
            else:
                raise ValueError(f"Invalid result: {result}")

        except Exception as e:
            exctype, value, tb = sys.exc_info()
            self.signals.error.emit((exctype, value, traceback.format_exc()))

    def stop(self):
        """Stop and clean the worker my sending signals downwards.

        DO NOT emit signals here. This is only for stopping the worker.
        """
        print("Worker: Stopping...", flush=True)
        self._stop_event.set()
        self.join()
