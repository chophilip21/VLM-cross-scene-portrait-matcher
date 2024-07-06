"""Divide functional and UI related logic."""

import photolink.utils.enums as enums
from photolink.utils.function import search_all_images, read_config
from PySide6.QtCore import Slot
from PySide6.QtWidgets import QLabel, QMessageBox
from photolink.pipeline.qss import *
import json
from photolink.pipeline.front import MainWindowFront, ProgressWidget
from photolink import get_application_path, get_config_file
from photolink.workers import Worker
from photolink.workers.jobs import JobProcessor
from pathlib import Path
import sys
import time
import threading

class MainWindow(MainWindowFront):
    """All functional codes related to Pyside go here."""

    def __init__(self):
       
        super().__init__()
        self.application_path = get_application_path()
        self.pipeline_path = self.application_path / "src" /"photolink" /"pipeline"
        config = get_config_file(self.application_path)
        self.config = read_config(config)
        self.venv_path = self.application_path / Path(self.config["WINDOWS"]["VIRTUAL_ENV"])
        self.job = {}
        self.all_stop = False
        self.operating_system = sys.platform
        print(f"Operating system: {self.operating_system}")
        self.drawUI()

        self.threads = []

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
            # clean up reference path text
            self.reference_path_selector.line_edit.setText("")
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
        self.init_time = time.time()
        self.log_message("Processing started.")

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

            self.preprocess_total = len(self.job["source"]) + len(self.job["reference"])

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
            self.preprocess_total = len(self.job["source"])

        else:
            self.change_button_status(True)
            raise ValueError("Invalid task selected")
        
        # Now passed all validation, so display the progress bar.
        self.progress_widget = ProgressWidget(self.stop_processing)
        self.progress_message_box = QMessageBox(self)
        self.progress_message_box.setWindowTitle("Processing has started. Please wait.")
        self.progress_message_box.setStandardButtons(QMessageBox.NoButton)
        self.progress_message_box.layout().addWidget(self.progress_widget)
        self.progress_message_box.setGeometry(500, 300, 450, 450)
        self.progress_message_box.show()
        self.progress_widget.setValue(0)

        # proceed to dump the job to a json file for worker nodes.
        job_json = self.cache_dir / "job.json"
        with open(job_json, "w") as f:
            json.dump(self.job, f)

        job = JobProcessor()
        worker = Worker(identifier=time.time())
        worker.signals.result.connect(self.task_result)
        worker.signals.progress.connect(self.task_progress)
        worker.signals.finished.connect(self.task_finished)
        worker.signals.error.connect(self.task_error)
        worker.start()
        self.threads.append(worker)
        print(f"Task started on thread: {threading.get_ident()}")

    def print_output(self, s):
        print(s)


    def process_finished(self):
        """Called when the Processing is finished."""
        self.change_button_status(True)

        if not self.all_stop:
            self.display_notification("Complete", "All operations completed successfully.")
            self.log_message("Processing finished.")
            self.stop_processing()
            self.progress_widget.setValue(100)

    def stop_processing(self):
        """Force stop the processing."""
        # self.threadpool.clear()
        # self.threadpool.waitForDone()
        self.progress_message_box.accept()
        self.process = None
        self.num_preprocessed = 0
        self.num_postprocessed = 0
        self.current_progress = 0
        self.preprocess_total = 0

    def update_progress(self, value):
        """Update the progress bar."""
        print(f"Progress: {value}")
        # if value > int(self.current_progress):
        #     self.current_progress = value
        #     self.progress_widget.setValue(self.current_progress)

 
    def task_result(self, result):
        print(f"Result: {result}")

    def task_progress(self, progress):
        print(f"Progress: {progress}%")

    def task_finished(self):
        print("Status: Task finished")

    def task_error(self, error):
        exctype, value, tb_str = error
        print('fuck')