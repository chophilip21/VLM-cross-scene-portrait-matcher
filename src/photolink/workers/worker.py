"""Execute the jobs for the worker via threading."""
from photolink.workers.jobs import JobProcessor
from PySide6.QtCore import Signal, QObject
import threading
from photolink.workers import WorkerSignals
import sys

class Worker(threading.Thread):
    def __init__(self, identifier):
        super().__init__()
        self.identifier = identifier
        self.signals = WorkerSignals()

    def run(self):
        print(f"Worker started on thread: {threading.get_ident()}")
        try:
            job = JobProcessor()
            job.run()
            # for i in range(1, 10):
            #     time.sleep(1)  # Simulate a long-running task
            #     progress = int((i / 5) * 100)
            #     print(f"Progress: {progress}%")
            #     self.signals.progress.emit(progress)
            # result = f"Task {self.identifier} completed"
            # self.signals.result.emit(result)
        except Exception as e:
            exctype, value, tb = sys.exc_info()
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        finally:
            self.signals.finished.emit()

