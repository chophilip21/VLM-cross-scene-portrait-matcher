"""Init file for worker module."""
from PySide6.QtCore import Signal, QObject
import traceback
import sys
import time
import threading
from photolink.workers.jobs import JobProcessor


class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    progress
        int indicating % progress

    '''
    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)
    progress = Signal(int)


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


# class Worker(threading.Thread):
#     '''
#     Worker thread

#     Inherits thread to handle worker thread setup, signals and wrap-up.

#     :param callback: The function callback to run on this worker thread. Supplied args and
#                      kwargs will be passed through to the runner.
#     :type callback: function
#     :param args: Arguments to pass to the callback function
#     :param kwargs: Keywords to pass to the callback function
#     '''

#     def __init__(self, fn, *args, **kwargs):
#         super(Worker, self).__init__()

#         # Store constructor arguments (re-used for processing)
#         self.fn = fn
#         self.args = args
#         self.kwargs = kwargs
#         self.signals = WorkerSignals()

#     @Slot()
#     def run(self):
#         '''
#         Initialise the runner function with passed args, kwargs.
#         '''
#         # Retrieve args/kwargs here; and fire processing using them
#         try:
#             result = self.fn(self, *self.args, **self.kwargs)
#         except:
#             traceback.print_exc()
#             exctype, value = sys.exc_info()[:2]
#             self.signals.error.emit((exctype, value, traceback.format_exc()))
#         else:
#             self.signals.result.emit(result)  # Return the result of the processing
#         finally:
#             self.signals.finished.emit()  # Done