"""Divide functional and UI related logic."""
import photolink.utils.enums as enums
from photolink.utils.function import search_all_images
from PySide6.QtCore import Slot
from PySide6.QtWidgets import QLabel
import os
from qss import *
import json
from PySide6.QtCore import QProcess, Signal, QTimer
from front import MainWindowFront


class MainWindow(MainWindowFront):

    progress_update = Signal(int)

    def __init__(self):
        """All functional codes go here."""
        super().__init__()
        self.process = None
        self.job = {}
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_progress)
        self.progress_update.connect(self.progress_bar.setValue)
        self.all_stop = False
        self.preprocess_ended = False

    @Slot()
    def handle_box_click(self):
        clicked_button = self.sender()
        task = clicked_button.findChild(QLabel).text()
        self.select_task(task)

    def select_task(self, task):
        if task == "Sample Match":
            self.instruction_label.setText(enums.Task.SAMPLE_MATCHING.value)
            self.reference_path_selector.line_edit.setPlaceholderText("")
            self.reference_path_selector.button.setEnabled(True)
            self.current_task = enums.Task.SAMPLE_MATCHING.name

        elif task == "Cluster":
            self.instruction_label.setText(enums.Task.CLUSTERING.value)
            self.reference_path_selector.line_edit.setPlaceholderText("Not required for clustering")
            self.reference_path_selector.button.setEnabled(False)
            self.current_task = enums.Task.CLUSTERING.name

        # Reset border colors for both boxes
        self.sample_match_box.setStyleSheet(
            f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {self.matching_color[0]}, stop:1 {self.matching_color[1]}); border: 2px solid black;"
        )
        self.cluster_box.setStyleSheet(
            f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {self.clustering_color[0]}, stop:1 {self.clustering_color[1]}); border: 2px solid black;"
        )

        # Highlight the selected box
        if task == "Sample Match":
            self.sample_match_box.setStyleSheet(
                self.sample_match_box.styleSheet() + " border: 2px solid white;"
            )
        elif task == "Cluster":
            self.cluster_box.setStyleSheet(
                self.cluster_box.styleSheet() + " border: 2px solid white;"
            )

    def process_jobs(self):
        """Call the multiprocessing method when the start button is clicked."""

        self.change_button_status(False)

        # first check there is output path.
        if not self.output_path_selector.line_edit.text():
            self.display_notification(
                enums.ErrorMessage.PATH_NOT_SELECTED.name,
                enums.ErrorMessage.PATH_NOT_SELECTED.value,
            )

            self.change_button_status(True)
            return

        self.job["output"] = self.output_path_selector.line_edit.text()

        # start by generating jobs based on the selected task
        if self.current_task == enums.Task.SAMPLE_MATCHING.name:

            if (
                not self.source_path_selector.line_edit.text()
                or not self.reference_path_selector.line_edit.text()
            ):
                self.display_notification(
                    enums.ErrorMessage.PATH_NOT_SELECTED.name,
                    enums.ErrorMessage.PATH_NOT_SELECTED.value,
                )
                self.change_button_status(True)
                return

            self.job["task"] = enums.Task.SAMPLE_MATCHING.name
            self.job["source"] = search_all_images(self.source_path_selector.line_edit.text())
            self.job["reference"] = search_all_images(self.reference_path_selector.line_edit.text())

        elif self.current_task == enums.Task.CLUSTERING.name:

            if not self.source_path_selector.line_edit.text():
                self.display_notification(
                    enums.ErrorMessage.PATH_NOT_SELECTED.name,
                    enums.ErrorMessage.PATH_NOT_SELECTED.value,
                )
                self.change_button_status(True)
                return

            self.job["task"] = enums.Task.CLUSTERING.name
            self.job["source"] = search_all_images(self.source_path_selector.line_edit.text())

        else:
            self.change_button_status(True)
            raise ValueError("Invalid task selected")

        # proceed to dump the job to a json file. Only then can the worker process it.
        job_json = os.path.join(self.cache_dir, "job.json")
        with open(job_json, "w") as f:
            json.dump(self.job, f)

        # Handle all jobs in a seperate process to prevent conflict with UI.
        if self.process is None:
            self.console.setText(f'Executing process for {self.job["task"]}')
            self.p = QProcess()
            self.p.readyReadStandardOutput.connect(self.handle_stdout)
            self.p.readyReadStandardError.connect(self.handle_stderr)
            self.p.stateChanged.connect(self.handle_state)
            self.p.finished.connect(self.process_finished)  # Connect to new method
            self.p.start("python3", ["jobs.py"])

            # use timer and check progress to update progress bar.
            self.timer.start(1000)

    def process_finished(self):
        """Called when the Processing is finished."""
        self.timer.stop()
        self.progress_bar.setValue(100)
        self.change_button_status(True)

        if not self.all_stop:
            self.display_notification("Complete", "All operations completed successfully.")
            self.log_message("Processing finished.")

    def handle_stderr(self):
        """Gracefully handle errors coming from process."""
        data = self.p.readAllStandardError()
        stderr = bytes(data).decode("utf8")
        print(stderr, end="")

        # ignore these ones. Only I should be able to see it. 
        for line in stderr.split("\n"):
            if line.startswith("Traceback"):
                return
            
            if line.startswith("Invalid SOS parameters for sequential JPEG"):
                return

        self.all_stop = True
        self.log_message(stderr)
        self.display_notification("Error has occured", stderr)

    def handle_stdout(self):
        data = self.p.readAllStandardOutput()
        stdout = bytes(data).decode("utf8")
        print(stdout, end="")

        # Check for postprocessing progress updates coming from worker.py
        for line in stdout.split("\n"):
            
            # turn off check_progress logic. 
            if line.startswith("Preprocessing ended"):
                self.preprocess_ended = True
                return

            if line.startswith("POSTPROCESS_PROGRESS:"):
                progress = int(line.split(":")[-1])
                self.progress_bar.setValue(progress)
                return

        self.log_message(stdout)

    def handle_state(self, state):
        states = {
            QProcess.NotRunning: "Finished",
            QProcess.Starting: "Starting",
            QProcess.Running: "Running",
        }
        state_name = states[state]
        print(f"Worker state: {state_name}")

    def check_progress(self):
        """Monitor the progress of processes running."""
        source_images = len(self.job["source"])
        source_cache_dir = os.path.join(self.cache_dir, "source")

        if not os.path.exists(source_cache_dir):
            raise FileNotFoundError(
                f"Source cache directory not created properly: {source_cache_dir}"
            )

        processed_files = len(
            [name for name in os.listdir(source_cache_dir) if name.endswith(".pkl")]
        )

        # Use this function to monitor progress up until 50% mark.
        if source_images > 0 and not self.preprocess_ended:
            progress = (processed_files / source_images) * 50
            self.progress_update.emit(int(progress))
