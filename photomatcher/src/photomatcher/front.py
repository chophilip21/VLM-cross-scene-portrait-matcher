"""Seperate some parts of the main app to make things less crowded."""

import os
import toga
from toga.style import Pack
from toga.style.pack import ROW, CENTER
import shutil
import photomatcher.enums as enums
import logging
from datetime import datetime

timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_path = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_path, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level to DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Set the log message format
    handlers=[
        logging.FileHandler(os.path.join(log_path, f"app_{timestamp}.log")),  # Log to a file
        logging.StreamHandler()  # Also log to console (optional)
    ]
)

class PhotoMatcherFrontEnd(toga.App):
    """Frontend for the photo matching application."""

    def __init__(self, formal_name=None):
        """Initialize the toga modules."""
        super().__init__(formal_name=formal_name)

        self.home = os.path.expanduser("~")
        self.num_processes = os.cpu_count()
        self.chunksize = os.getenv("CHUNKSIZE", 10)
        self.top_n_face = int(os.getenv("TOP_N_FACE", 3))

        # set up cache path.
        self.source_cache = os.path.join(os.path.dirname(__file__), "cache/source")
        self.reference_cache = os.path.join(
            os.path.dirname(__file__), "cache/reference"
        )
        self.source_list_images = None
        self.reference_list_images = None
        self.debugger = logging.getLogger(__name__)

    def setup_cache_dir(self):
        """clean up cache folders on start up, and recreate dir"""

        if os.path.exists(self.source_cache):
            shutil.rmtree(self.source_cache)

        if os.path.exists(self.reference_cache):
            shutil.rmtree(self.reference_cache)

        os.makedirs(self.source_cache, exist_ok=True)
        os.makedirs(self.reference_cache, exist_ok=True)
        self.debugger.info("refreshing cache")

    def refresh_inputs(self, widget):
        """Clear all text inputs and reset progress bar."""
        self.src_path_input.value = ""
        self.ref_path_input.value = ""
        self.output_path_input.value = ""
        self.progress_bar.value = 0
        self.console_log.value = ""
        self.setup_cache_dir()
        self.display_console_message("Inputs refreshed")

        if self.task_selection.value == enums.Task.SAMPLE_MATCHING.value:
            self.display_console_message(enums.StatusLogMessage.SAMPLE_MATCHING.value)
        elif self.task_selection.value == enums.Task.CLUSTERING.value:
            self.display_console_message(enums.StatusLogMessage.CLUSTERING.value)
        else:
            raise NotImplementedError(
                f"Task {self.task_selection.value} not implemented."
            )

    def create_path_box(self, label_text, on_press_handler):
        """Create a box with a label, text input, and button to select a path."""
        path_box = toga.Box(style=Pack(direction=ROW, padding=5, alignment=CENTER))
        path_label = toga.Label(label_text, style=Pack(padding=(0, 5)))
        path_input = toga.TextInput(readonly=True, style=Pack(flex=1))
        path_button = toga.Button(
            "Choose...", on_press=on_press_handler, style=Pack(padding=5)
        )
        path_box.add(path_label)
        path_box.add(path_input)
        path_box.add(path_button)
        return path_input, path_box

    def update_visibility(self, widget=None):
        """Update visibility of UI components based on selected task."""
        if self.task_selection.value == enums.Task.SAMPLE_MATCHING.value:
            if self.ref_path_box[1] not in self.main_box.children:
                self.main_box.insert(self.ref_path_box_index, self.ref_path_box[1])
                self.console_log.value = ""
                self.display_console_message(enums.StatusLogMessage.START.value)
        elif self.task_selection.value == enums.Task.CLUSTERING.value:
            if self.ref_path_box[1] in self.main_box.children:
                self.main_box.remove(self.ref_path_box[1])
                self.console_log.value = ""
                self.display_console_message(enums.StatusLogMessage.CLUSTERING.value)

    async def select_src_path(self, widget):
        """Select the source images folder."""
        await self.select_path(self.src_path_input, "Source Images Folder")

    async def select_ref_path(self, widget):
        """Select the reference images folder."""
        await self.select_path(self.ref_path_input, "Reference Images Folder")

    async def select_output_path(self, widget):
        """Select the output folder."""
        await self.select_path(self.output_path_input, "Output Folder")

    async def select_path(self, input_widget, dialog_title):
        """Select the input paths using a dialog."""
        try:
            result = await self.main_window.select_folder_dialog(
                dialog_title, initial_directory=self.home
            )
            if result:
                input_widget.value = result
                self.display_console_message(f"{dialog_title} selected: {result}")
            else:
                input_widget.value = "No folder selected!"
                self.display_console_message(f"{dialog_title} selection canceled")
        except Exception as e:
            input_widget.value = "Error selecting folder!"
            self.display_console_message(f"Error selecting folder: {e}")
