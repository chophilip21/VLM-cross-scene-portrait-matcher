"""Execute the jobs for the worker via threading."""
from photolink.workers.jobs import JobProcessor
import threading
import multiprocessing as mp
from photolink.workers import WorkerSignals
import photolink.utils.enums as enums
import sys
import traceback

class Worker(threading.Thread):
    """Execute the jobs for the worker via threading to prevent GUI freeze. Calls the JobProcessor to run the jobs."""
    def __init__(self, identifier):
        super().__init__()
        self.identifier = identifier
        self.signals = WorkerSignals()

        # Use this to send signals to the worker.
        self._stop_event = mp.Event()

    def run(self):
        try:
            job = JobProcessor(stop_event=self._stop_event)
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
        finally:
            self.signals.finished.emit()

    def stop(self):
        """Stop the worker my sending signals downwards."""
        print("Worker: Stopping...", flush=True)
        self._stop_event.set()       
        self.signals.finished.emit()
        self.join()