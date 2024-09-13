"""Init file for worker module."""

from PySide6.QtCore import QObject, Signal


class WorkerSignals(QObject):
    """
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

    """

    finished = Signal()
    stopped = Signal()
    error = Signal(tuple)
    result = Signal(object)
    progress = Signal(int)
