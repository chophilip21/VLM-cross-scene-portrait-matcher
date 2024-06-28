"""Divide functional and UI related logic."""

import photolink.utils.enums as enums
from photolink.utils.function import search_all_images, read_config
from PySide6.QtCore import Slot, QProcessEnvironment, QProcess, QDir
from PySide6.QtWidgets import QLabel, QMessageBox
from photolink.pipeline.qss import *
import json
from photolink.pipeline.front import MainWindowFront, ProgressWidget
from photolink.pipeline.main import get_application_path, get_config_file
from pathlib import Path
import sys
import time

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
            self.console.setText(f'Executing process for {self.job["task"]}')
            self.process = QProcess()
            self.process.readyReadStandardOutput.connect(self.handle_stdout)
            self.process.readyReadStandardError.connect(self.handle_stderr)
            self.process.stateChanged.connect(self.handle_state)
            self.process.finished.connect(self.process_finished)  # Connect to new method

            # run jobs.py as subprocess.
            job_script_path = self.pipeline_path / Path("jobs.py")
            job_script_directory = job_script_path.parent
            self.process.setWorkingDirectory(str(job_script_directory))

            # env to make sure it uses right Python.
            env = QProcessEnvironment.systemEnvironment()

            # Windows need special handling for venv and path
            if self.operating_system == enums.OperatingSystem.WINDOWS.value:
                python_executable = self.venv_path / "Scripts" / "python.exe"

                if not python_executable.exists():
                    raise FileNotFoundError(f"Python executable not found at {python_executable}")
                
                # We need to insert the venv path into the environment variables
                env.insert("PYTHONHOME", str(self.venv_path))   
                env.insert("PYTHONPATH", str(self.venv_path / "Lib" / "site-packages"))
                env.insert("PATH", str(self.venv_path / "Scripts") + ";" + env.value("PATH"))
                self.process.setProcessEnvironment(env)

                # execute
                native_job_script_path = QDir.toNativeSeparators(str(job_script_path))
                self.process.start(str(python_executable), [native_job_script_path])

            # Linux is more straightforward
            elif self.operating_system == enums.OperatingSystem.LINUX.value:
                native_job_script_path = job_script_path

                python_executable = self.venv_path / "bin" / "python3"

                if not python_executable.exists():
                    raise FileNotFoundError(f"Python executable not found at {python_executable}")

                # TODO: Do not hard code Python version like this.
                python_version = self.config["VERSION"]["PYTHON_VERSION"] 
                # We need to insert the venv path into the environment variables
                env.insert("PYTHONHOME", str(self.venv_path))
                env.insert("PYTHONPATH", str(self.venv_path / "lib" / f"{python_version}" / "site-packages"))
                env.insert("PATH", str(self.venv_path / "bin") + ":" + env.value("PATH"))
                self.process.setProcessEnvironment(env)

                # execute the process
                self.process.start(python_executable, [native_job_script_path])

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

