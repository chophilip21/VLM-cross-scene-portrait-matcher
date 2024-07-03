"""Divide functional and UI related logic."""

import photolink.utils.enums as enums
from photolink.utils.function import search_all_images, read_config
from PySide6.QtCore import Slot, QProcess, QDir
from PySide6.QtWidgets import QLabel, QMessageBox
from photolink.pipeline.qss import *
import json
from photolink.pipeline.front import MainWindowFront, ProgressWidget
from photolink import get_application_path, get_config_file
from pathlib import Path
import sys
import time
from photolink.server.service import ServerThread
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Qt
from PySide6.QtGui import QMovie, QFont

class MainWindow(MainWindowFront):
    """All functional codes related to Pyside go here."""

    def __init__(self):
       
        super().__init__()
        self.process = None
        self.application_path = get_application_path()
        self.pipeline_path = self.application_path / "src" /"photolink" /"pipeline"
        config = get_config_file(self.application_path)
        self.config = read_config(config)
        self.venv_path = self.application_path / Path(self.config["WINDOWS"]["VIRTUAL_ENV"])
        self.job = {}
        self.all_stop = False
        self.operating_system = sys.platform
        print(f"Operating system: {self.operating_system}")

        # progress bar monitoring purposes
        self.current_progress = 0
        self.preprocess_total = 0
        self.num_preprocessed = 0
        self.num_postprocessed = 0

        # Set default window size
        self.setWindowTitle("PhotoMatcher v.0.01")
        self.setGeometry(500, 300, 800, 600)

        # need to start server here before drawing anything.
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.loading_label = QLabel('Disclaimer: This software is only meant for internal usage.', self)
        self.loading_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(15)
        self.loading_label.setFont(font)

        # add spinner to the loading label
        self.spinner = QLabel(self)
        self.spinner.setAlignment(Qt.AlignCenter)
        self.loading_gif = str(self.application_path / Path(self.config.get("IMAGES", "LOAD_GIF")))
        self.movie = QMovie(self.loading_gif)
        self.spinner.setMovie(self.movie)
        self.movie.start()

        self.layout.addStretch(1)
        self.layout.addWidget(self.spinner)
        self.layout.addWidget(self.loading_label)
        self.layout.addStretch(1)

        # start server
        self.server_thread = ServerThread()
        self.server_thread.server_ready.connect(self.on_server_ready)
        self.server_thread.start()

    def on_server_ready(self, ready):
        """When server responds, draw the main UI or show error message."""
        if ready:
            self.movie.stop()
            self.drawUI()
        else:
            self.loading_label.setText("Failed to start server. Please try again.")

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

        # Handle all jobs in a seperate process to prevent conflict with UI.
        if self.process is None:
            self.log_message("Running Interpreter checks.")
            self.console.setText(f'Executing process for {self.job["task"]}')
            self.process = QProcess()
            self.process.readyReadStandardOutput.connect(self.handle_stdout)
            self.process.readyReadStandardError.connect(self.handle_stderr)
            self.process.stateChanged.connect(self.handle_state)
            self.process.finished.connect(self.process_finished)  

            # run jobs.py as subprocess.
            job_script_path = self.pipeline_path / Path("jobs.py")
            job_script_directory = job_script_path.parent
            self.process.setWorkingDirectory(str(job_script_directory))

            # Windows need special handling for venv and path
            if self.operating_system == enums.OperatingSystem.WINDOWS.value:

                # # python_executable = self.venv_path / "Scripts" / "python.exe"
                # python_executable = sys.executable
                # base_path = sys._MEIPASS if hasattr(sys, '_MEIPASS') else None
                # self.log_message(f"Python executable: {python_executable}, base path: {base_path}")

                # if not Path(python_executable).exists():
                #     raise FileNotFoundError(f"Python executable not found at {python_executable}")
         
                native_job_script_path = QDir.toNativeSeparators(str(job_script_path))

                self.process.start('python3', [native_job_script_path])

            # Linux is more straightforward
            elif self.operating_system == enums.OperatingSystem.LINUX.value:
                native_job_script_path = job_script_path


                # python_executable = self.venv_path / "bin" / "python3"

                # python_executable = sys.executable
                
                # if not Path(python_executable.exists()):
                #     raise FileNotFoundError(f"Python executable not found at {python_executable}")
        
                # self.log_message(f"Python executable: {python_executable}")

                # execute the process
                self.process.start(sys.executable, [native_job_script_path])

            else:
                raise NotImplementedError(
                    f"Operating system not supported: {self.operating_system}"
                )

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
        if self.process is not None:
            self.process.kill()

        self.progress_message_box.accept()
        self.process = None
        self.num_preprocessed = 0
        self.num_postprocessed = 0
        self.current_progress = 0
        self.preprocess_total = 0

    def update_progress(self, value):
        """Update the progress bar."""
        if value > int(self.current_progress):
            self.current_progress = value
            self.progress_widget.setValue(self.current_progress)

    def handle_stderr(self):
        """Gracefully handle errors coming from process."""
        data = self.process.readAllStandardError()
        stderr = bytes(data).decode("utf8")
        print(stderr, end="")

        # ignore these ones. Only I should be able to see it.
        for l in stderr.split("\n"):

            if l.startswith("Invalid SOS parameters for sequential JPEG"):
                return

        self.all_stop = True
        self.log_message(stderr)
        self.display_notification("Error has occured", stderr)
        self.stop_processing()

    def handle_stdout(self):
        """Use this to update progress bar and log messages."""
        data = self.process.readAllStandardOutput()
        stdout = bytes(data).decode("utf8")
        print(stdout, end="")

        # Check for postprocessing progress updates coming from worker.py
        for line in stdout.split("\n"):
            
            # update first 50% of the progress bar
            if line.startswith("Pre-Processing:"):
                progress = int((self.num_preprocessed / self.preprocess_total) * 50)
                self.update_progress(progress)
                self.num_preprocessed += 1
                return

            # update later 50% of the progress bar
            if line.startswith("Post-Processing:"):
                progress = int(line.split(":")[-1].strip())
                self.update_progress(progress)
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
