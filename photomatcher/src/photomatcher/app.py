"""Combines TOGO frontend app code, calling ml model code from worker.py"""

import os
import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW, CENTER
import photomatcher.enums as enums
import photomatcher.worker as worker
import photomatcher.utils as utils
from photomatcher.front import PhotoMatcherFrontEnd
import asyncio
import multiprocessing

class PhotoMatcher(PhotoMatcherFrontEnd):
    """Photo matching main application."""

    def __init__(self, formal_name="Photo Matcher"):
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
            on_press=self.execute,
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
        self.display_console_message(enums.StatusLogMessage.START.value)
        self.update_visibility()

    def display_console_message(self, message):
        """Append a message to the console log."""
        self.console_log.value += message + "\n"
        self.debugger.info(message)

    async def execute(self, widget):
        """Run the photo matching processing."""
        self.source_path = self.src_path_input.value
        self.reference_path = self.ref_path_input.value
        self.output_path = self.output_path_input.value
        self.fail_path = self.output_path_input.value + "/uncertain"

        if self.task_selection.value == enums.Task.SAMPLE_MATCHING.value:
            self.debugger.info("Running sample matching.")
            if not all([self.source_path, self.reference_path, self.output_path]):
                self.main_window.error_dialog(
                    "Invalid Command", enums.ErrorMessage.PATH_NOT_SELECTED.value
                )
                self.debugger.error("source, reference, or output path not selected.")
                return
        else:
            self.debugger.info("Running clustering.")
            if not all([self.source_path, self.output_path]):
                self.main_window.error_dialog(
                    "Invalid Command", enums.ErrorMessage.PATH_NOT_SELECTED.value
                )
                self.debugger.error("source or output path not selected.")
                return

        os.makedirs(self.fail_path, exist_ok=True)
        self.display_console_message("Starting pre-processing step...")
        loop = asyncio.get_event_loop()
        receipt = await loop.run_in_executor(None, self._threaded_cpu_tasks)

        # if none returned, do not proceed.
        if receipt is None:
            self.display_console_message("pre-processing failed. Check the logs again.")
            return

        # Run rest of the tasks in the main thread
        if self.task_selection.value == enums.Task.SAMPLE_MATCHING.value:
            final_result = worker.match_embeddings(**receipt)
            self.debugger.info("postprocess for matching finished.")
        elif self.task_selection.value == enums.Task.CLUSTERING.value:
            final_result = worker.cluster_embeddings(**receipt)
            self.debugger.info("postprocess for clustering finished.")
            
        if "error" in final_result:
            self.main_window.error_dialog(
                "Post-processing Failed",
                "An error occurred during matching embeddings. Please check the logs.",
            )
            self.debugger.error(final_result["error"])
            self.progress_bar.stop()
            return

        if "missed_count" in final_result:
            self.display_console_message(
                f"Unable to cluster {final_result['missed_count']} faces in the output. Saving them."
            )

        self.progress_bar.value = 100
        self.progress_bar.stop()
        self.main_window.info_dialog(
            "Completed", "Your photo matching processed successfully!"
        )
        self.display_console_message("Processing completed.")

    def _threaded_cpu_tasks(self):
        """Run common, cpu-intensive ML models (face detection, embedding conversion) here firs."""
        self.progress_bar.start()
        self.progress_bar.value = 10

        if self.task_selection.value == enums.Task.SAMPLE_MATCHING.value:
            result = self.preprocess_sample_matching()
            self.debugger.info("preprocess for sample matching done")
        elif self.task_selection.value == enums.Task.CLUSTERING.value:
            result = self.preprocess_clustering()
            self.debugger.info("preprocess for clustering done")
        else:
            raise NotImplementedError(
                f"Task {self.task_selection.value} not implemented."
            )

        if "error" in result:
            self.main_window.error_dialog(
                "Processing Failed",
                "An error occurred during processing. Please check the logs.",
            )
            self.debugger.error(result["error"])
            self.progress_bar.stop()
            return None

        return result

    def preprocess_sample_matching(self) -> dict:
        """Preprocessing for the matching algorithm."""
        self.source_list_images = utils.search_all_images(self.source_path)

        if len(self.source_list_images) == 0:
            self.main_window.error_dialog(
                "Invalid Command", enums.ErrorMessage.SOURCE_FOLDER_EMPTY.value
            )
            self.debugger.error("source folder empty, cannot run sample matching.")
            return

        self.reference_list_images = utils.search_all_images(self.reference_path)

        if len(self.reference_list_images) == 0:
            self.main_window.error_dialog(
                "Invalid Command", enums.ErrorMessage.REFERENCE_FOLDER_EMPTY.value
            )
            self.debugger.error("reference folder empty, cannot run sample matching.")
            return

        self.progress_bar.value = 25
        self.display_console_message(
            f"Processing {len(self.source_list_images)} source images."
        )

        worker.run_model_mp(
            self.source_list_images,
            self.num_processes,
            self.chunksize,
            self.source_cache,
            self.fail_path,
            self.top_n_face,
        )

        self.display_console_message(
            f"Processing {len(self.reference_list_images)} reference images."
        )

        self.progress_bar.value = 50
        worker.run_model_mp(
            self.reference_list_images,
            self.num_processes,
            self.chunksize,
            self.reference_cache,
            self.fail_path,
            self.top_n_face,
        )
        self.progress_bar.value = 75

        self.display_console_message(
            "Conversion completed. Now matching and saving results."
        )

        inputs = {
            "source_cache": self.source_cache,
            "reference_cache": self.reference_cache,
            "source_list_images": self.source_list_images,
            "reference_list_images": self.reference_list_images,
            "output_path": self.output_path,
        }

        return inputs

    def preprocess_clustering(self) -> dict:
        """Preprocessing for the clustering algorithm."""
        self.source_list_images = utils.search_all_images(self.source_path)

        if len(self.source_list_images) == 0:
            self.main_window.error_dialog(
                "Invalid Command", enums.ErrorMessage.SOURCE_FOLDER_EMPTY.value
            )
            return

        self.progress_bar.value = 25
        self.display_console_message(
            f"Processing {len(self.source_list_images)} source images."
        )

        worker.run_model_mp(
            self.source_list_images,
            self.num_processes,
            self.chunksize,
            self.source_cache,
            self.fail_path,
        )
        self.progress_bar.value = 50
        self.display_console_message(
            "Embedding conversion completed. Now Clustering the results."
        )

        # HDBSCAN outperforms DBSCAN and OPTICS in most cases.
        inputs = {
            "source_cache": self.source_cache,
            "source_list_images": self.source_list_images,
            "clustering_algorithm": enums.ClusteringAlgorithm.HDBSCAN.value,
            "eps": 0.5,
            "min_samples": 2,
            "output_path": self.output_path,
            "fail_path": self.fail_path,
        }

        return inputs


def main():
    """Main entry point for the application."""
    multiprocessing.freeze_support() # required for windows
    return PhotoMatcher()