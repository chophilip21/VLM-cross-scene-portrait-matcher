"""Combines TOGO frontend app code, calling ml model code from worker.py"""

import os
import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW, CENTER
import logging
import photomatcher.enums as enums
import photomatcher.worker as worker
import photomatcher.utils as utils
from photomatcher.front import PhotoMatcherFrontEnd
import asyncio

class PhotoMatcher(PhotoMatcherFrontEnd):
    """Photo matching main application."""

    def __init__(self, formal_name='Photo Matcher'):
        """Initialize the toga modules."""
        super().__init__(formal_name=formal_name)
 
    def startup(self):
        """Create the main window for the application."""

        self.setup_cache_dir()

        # Application main box. Modules will be added to this box, from top to bottom.
        self.main_box = toga.Box(
            style=Pack(direction=COLUMN, padding=10, alignment=CENTER)
        )

        # Logo module
        logo_path = os.path.join(os.path.dirname(__file__), "resources/logo.jpg")
        logo = toga.Image(logo_path)
        logo_view = toga.ImageView(
            logo, style=Pack(width=300, height=300, alignment=CENTER)
        )

        # Task Selection
        task_label = toga.Label("Task:", style=Pack(padding=(0, 5)))
        self.task_selection = toga.Selection(
            items=[
                enums.Task.SAMPLE_MATCHING.value,
                enums.Task.CLUSTERING.value,
            ],
            on_change=self.update_visibility,
            style=Pack(flex=1, padding=5),
        )
        task_box = toga.Box(style=Pack(direction=ROW, padding=5, alignment=CENTER))
        task_box.add(task_label)
        task_box.add(self.task_selection)

        # Source Images Path
        self.src_path_box = self.create_path_box(
            "Source img path:", self.select_src_path
        )
        self.src_path_input = self.src_path_box[0]

        # Reference Images Path
        self.ref_path_box = self.create_path_box(
            "Reference img Path:", self.select_ref_path
        )
        self.ref_path_input = self.ref_path_box[0]

        # Output Path
        output_path_box = self.create_path_box("Output path:", self.select_output_path)
        self.output_path_input = output_path_box[0]

        # Buttons Box
        buttons_box = toga.Box(style=Pack(direction=ROW, padding=5, alignment=CENTER))
        self.run_button = toga.Button(
            "Run Processing",
            on_press=self.run_processing,
            style=Pack(padding=5, background_color="#28a745", color="white", width=200),
        )
        self.refresh_button = toga.Button(
            "Refresh",
            on_press=self.refresh_inputs,
            style=Pack(padding=5, background_color="#3545dc", color="white", width=200),
        )
        buttons_box.add(self.run_button)
        buttons_box.add(self.refresh_button)

        # Progress Bar
        self.progress_bar = toga.ProgressBar(
            max=100, value=0, style=Pack(padding=10, width=1000, alignment=CENTER)
        )

        # Console Log Box
        self.console_log = toga.MultilineTextInput(
            readonly=True,
            style=Pack(
                flex=1,
                padding=10,
                background_color="black",
                color="white",
                height=150,
                alignment=CENTER,
            ),
        )

        # Adding all components to the main box
        self.main_box.add(logo_view)
        self.main_box.add(task_box)
        self.main_box.add(self.src_path_box[1])
        self.ref_path_box_index = len(self.main_box.children)
        self.main_box.add(self.ref_path_box[1])
        self.main_box.add(output_path_box[1])
        self.main_box.add(buttons_box)
        self.main_box.add(self.progress_bar)
        self.main_box.add(self.console_log)
        self.main_window = toga.MainWindow(title=self.formal_name)
        self.main_window.content = self.main_box
        self.main_window.show()
        self.log_message(enums.StatusLogMessage.START.value)
        self.update_visibility()

    def log_message(self, message):
        """Append a message to the console log."""
        self.console_log.value += message + "\n"

    async def run_processing(self, widget):
        """Run the photo matching processing."""
        self.source_path = self.src_path_input.value
        self.reference_path = self.ref_path_input.value
        self.output_path = self.output_path_input.value
        self.fail_path = self.output_path_input.value + "/uncertain"

        if self.task_selection.value == enums.Task.SAMPLE_MATCHING.value:
            if not all([self.source_path, self.reference_path, self.output_path]):
                self.main_window.error_dialog(
                    "Invalid Command", enums.ErrorMessage.PATH_NOT_SELECTED.value
                )
                return
        else:
            if not all([self.source_path, self.output_path]):
                self.main_window.error_dialog(
                    "Invalid Command", enums.ErrorMessage.PATH_NOT_SELECTED.value
                )
                return

        os.makedirs(self.fail_path, exist_ok=True)
        self.log_message("Starting processing...")
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._run_processing)

        # if none returned, do not proceed.
        if result is None:
            self.log_message("Processing failed. Check the inputs again.")
            return

        self.main_window.info_dialog(
            "Processing Completed", "Your photo matching processed successfully!"
        )
        self.log_message("Processing completed.")

    def _run_processing(self):
        """Run ML models here."""
        self.progress_bar.start()
        self.progress_bar.value = 10

        if self.task_selection.value == enums.Task.SAMPLE_MATCHING.value:
            result = self.run_sample_matching()
            print("sample matching done")
        elif self.task_selection.value == enums.Task.CLUSTERING.value:
            result = self.run_clustering()
            print("clustering done")
        else:
            raise NotImplementedError(
                f"Task {self.task_selection.value} not implemented."
            )

        if "error" in result:
            self.main_window.error_dialog("Processing Failed", result["error"])
            self.progress_bar.stop()
            return False

        self.progress_bar.value = 100
        self.progress_bar.stop()

        return True

    def run_sample_matching(self) -> dict:
        """Run the matching algorithm."""
        self.source_list_images = utils.search_all_images(self.source_path)

        if len(self.source_list_images) == 0:
            self.main_window.error_dialog(
                "Invalid Command", enums.ErrorMessage.SOURCE_FOLDER_EMPTY.value
            )
            return

        self.reference_list_images = utils.search_all_images(self.reference_path)

        if len(self.reference_list_images) == 0:
            self.main_window.error_dialog(
                "Invalid Command", enums.ErrorMessage.REFERENCE_FOLDER_EMPTY.value
            )
            return

        self.progress_bar.value = 25
        self.log_message(f"Processing {len(self.source_list_images)} source images.")

        worker.run_model_mp(
            self.source_list_images,
            self.num_processes,
            self.chunksize,
            self.source_cache,
            self.fail_path,
        )

        self.log_message(
            f"Processing {len(self.reference_list_images)} reference images."
        )

        self.progress_bar.value = 50
        worker.run_model_mp(
            self.reference_list_images,
            self.num_processes,
            self.chunksize,
            self.reference_cache,
            self.fail_path,
        )
        self.progress_bar.value = 75

        self.log_message(
            "Embedding conversion completed. Now matching and saving results."
        )

        inputs = {
            "source_cache": self.source_cache,
            "reference_cache": self.reference_cache,
            "source_list_images": self.source_list_images,
            "reference_list_images": self.reference_list_images,
            "output_path": self.output_path,
        }

        matching_result = worker.match_embeddings(**inputs)

        return matching_result

    def run_clustering(self) -> dict:
        """Run the clustering algorithm."""
        self.source_list_images = utils.search_all_images(self.source_path)

        if len(self.source_list_images) == 0:
            self.main_window.error_dialog(
                "Invalid Command", enums.ErrorMessage.SOURCE_FOLDER_EMPTY.value
            )
            return

        self.progress_bar.value = 25
        self.log_message(f"Processing {len(self.source_list_images)} source images.")

        worker.run_model_mp(
            self.source_list_images,
            self.num_processes,
            self.chunksize,
            self.source_cache,
            self.fail_path,
        )
        self.progress_bar.value = 50
        self.log_message("Embedding conversion completed. Now Clustering the results.")

        # HDBSCAN outperforms DBSCAN and OPTICS in most cases.
        inputs = {
            "source_cache": self.source_cache,
            "source_list_images": self.source_list_images,
            "clustering_algorithm": enums.ClusteringAlgorithm.HDBSCAN.value,
            "eps": 0.5,
            "min_samples": 3,
            "output_path": self.output_path,
            "fail_path": self.fail_path,
        }

        cluster_result = worker.cluster_embeddings(**inputs)

        return cluster_result


def main():
    return PhotoMatcher()
