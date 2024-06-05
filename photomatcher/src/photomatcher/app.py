import os
from dotenv import load_dotenv

env_file = os.path.join(os.path.dirname(__file__), "resources/config.env")
load_dotenv(env_file)

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW, CENTER
import logging
from photomatcher.enums import IMAGE_EXTENSION
from photomatcher.worker import run_model_mp
import asyncio
import shutil
import faiss
import pickle
import numpy as np


class PhotoMatcher(toga.App):
    """Frontend for the photo matching application."""

    def __init__(self, formal_name=None):
        """Initialize the toga modules."""
        super().__init__(formal_name=formal_name)
        self.home = os.path.expanduser("~")
        self.num_processes = os.cpu_count()
        self.chunksize = os.getenv("CHUNKSIZE", 10)

        # set up cache path.
        self.source_cache = os.path.join(os.path.dirname(__file__), "cache/source")
        self.reference_cache = os.path.join(
            os.path.dirname(__file__), "cache/reference"
        )
        self.source_list_images = None
        self.reference_list_images = None

    def setup_cache_dir(self):
        """clean up cache folders on start up, and recreate dir"""

        if os.path.exists(self.source_cache):
            shutil.rmtree(self.source_cache)

        if os.path.exists(self.reference_cache):
            shutil.rmtree(self.reference_cache)

        os.makedirs(self.source_cache, exist_ok=True)
        os.makedirs(self.reference_cache, exist_ok=True)
        print('refreshing cache')

    def startup(self):
        """Create the main window for the application."""

        self.setup_cache_dir()

        main_box = toga.Box(style=Pack(direction=COLUMN, padding=10, alignment=CENTER))
        logo_path = os.path.join(os.path.dirname(__file__), "resources/logo.jpg")
        if not os.path.isfile(logo_path):
            raise RuntimeError(f"Logo file not found: {logo_path}")

        logo = toga.Image(logo_path)
        logo_view = toga.ImageView(
            logo, style=Pack(width=300, height=300, alignment=CENTER)
        )

        # Source Images Path
        source_path_box = self.create_path_box(
            "Source Images Path:", self.select_source_path
        )
        self.source_path_input = source_path_box[0]

        # Reference Images Path
        reference_path_box = self.create_path_box(
            "Reference Images Path:", self.select_reference_path
        )
        self.reference_path_input = reference_path_box[0]

        # Output Path
        output_path_box = self.create_path_box("Output Path:", self.select_output_path)
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
        main_box.add(logo_view)
        main_box.add(source_path_box[1])
        main_box.add(reference_path_box[1])
        main_box.add(output_path_box[1])
        main_box.add(buttons_box)
        main_box.add(self.progress_bar)
        main_box.add(self.console_log)

        self.log_message(
            "Welcome to Photo Matcher. Start by selecting the source, reference, and output folders above."
        )

        self.main_window = toga.MainWindow(title=self.formal_name)
        self.main_window.content = main_box
        self.main_window.show()

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

    async def select_source_path(self, widget):
        """Select the source images folder."""
        await self.select_path(self.source_path_input, "Source Images Folder")

    async def select_reference_path(self, widget):
        """Select the reference images folder."""
        await self.select_path(self.reference_path_input, "Reference Images Folder")

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
                self.log_message(f"{dialog_title} selected: {result}")
            else:
                input_widget.value = "No folder selected!"
                self.log_message(f"{dialog_title} selection canceled")
        except Exception as e:
            logging.error(f"Error selecting path: {e}")
            input_widget.value = "Error selecting folder!"
            self.log_message(f"Error selecting folder: {e}")

    def refresh_inputs(self, widget):
        """Clear all text inputs and reset progress bar."""
        self.source_path_input.value = ""
        self.reference_path_input.value = ""
        self.output_path_input.value = ""
        self.progress_bar.value = 0
        self.console_log.value = ""
        self.setup_cache_dir()
        self.log_message(
            "Inputs refreshed. Start again by selecting the source, reference, and output folders above."
        )

    def log_message(self, message):
        """Append a message to the console log."""
        self.console_log.value += message + "\n"

    async def run_processing(self, widget):
        """Run the photo matching processing."""
        self.source_path = self.source_path_input.value
        self.reference_path = self.reference_path_input.value
        self.output_path = self.output_path_input.value
 
        if not all([self.source_path, self.reference_path, self.output_path]):
            self.main_window.error_dialog(
                "Error",
                "Please select all required folders before running the processing.",
            )
            return

        if (
            len(os.listdir(self.source_path)) == 0
            or len(os.listdir(self.reference_path)) == 0
        ):
            self.main_window.error_dialog(
                "Error", "Source and Reference folders must not be empty."
            )
            return

        self.fail_path = self.output_path_input.value + "/missed"
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

        self.source_list_images = [
            os.path.join(self.source_path, file)
            for file in os.listdir(self.source_path)
            if file.split(".")[-1] in IMAGE_EXTENSION
        ]

        if len(self.source_list_images) == 0:
            self.main_window.error_dialog(
                "Error",
                "Please make sure that there are image files in the source folder.",
            )
            return

        self.reference_list_images = [
            os.path.join(self.reference_path, file)
            for file in os.listdir(self.reference_path)
            if file.split(".")[-1] in IMAGE_EXTENSION
        ]

        if len(self.reference_list_images) == 0:
            self.main_window.error_dialog(
                "Error",
                "Please make sure that there are image files in the reference folder.",
            )
            return

        self.progress_bar.value = 25

        # Run the model for source
        self.log_message(f"Processing {len(self.source_list_images)} source images.")

        run_model_mp(
            self.source_list_images, self.num_processes, self.chunksize, self.source_cache, self.fail_path
        )

        self.log_message(f"Processing {len(self.reference_list_images)} reference images.")

        self.progress_bar.value = 50
        run_model_mp(
            self.reference_list_images,
            self.num_processes,
            self.chunksize,
            self.reference_cache,
            self.fail_path,
        )
        self.progress_bar.value = 75

        self.log_message("Embedding conversion completed. Now matching and saving results.")

        self.match_embeddings()

        self.progress_bar.value = 100
        self.progress_bar.stop()

        return True


    def match_embeddings(self):
        """Match the embeddings and save the match."""
    
        faiss_index = faiss.IndexFlatL2(128)
        source_embeddings = [os.path.join(self.source_cache, file) for file in os.listdir(self.source_cache) if file.split(".")[-1] == "pkl"]

        reference_embeddings = [os.path.join(self.reference_cache, file) for file in os.listdir(self.reference_cache) if file.split(".")[-1] == "pkl"]

        # Create quick look up table for the source and reference images.
        source_dict = {file.split("/")[-1].split(".")[0]: file for file in self.source_list_images}
        reference_dict = {file.split("/")[-1].split(".")[0]: file for file in self.reference_list_images}


        # read the pickle files from the source embedding_path, and add it to the faiss index.
        for file in source_embeddings:
            with open(os.path.join(self.source_cache, file), "rb") as f:
                embedding = pickle.load(f)

                if isinstance(embedding, list):
                    embedding = np.array(embedding)
                else:
                    raise ValueError(f"Error on {file}. Embedding must be a list, not {type(embedding)}")

                if embedding.shape != (1, 128):
                    raise ValueError(f"Error on {file}. Embedding must be a (1, 128) numpy array, not {embedding.shape}")

                faiss_index.add(embedding)

        print('added all the source embeddings to faiss index...')

        for file in reference_embeddings:
            with open(os.path.join(self.reference_cache, file), "rb") as f:
                embedding = pickle.load(f)

                if isinstance(embedding, list):
                    embedding = np.array(embedding)
                else:
                    raise ValueError(f"Error on {file}. Embedding must be a list, not {type(embedding)}")

                if embedding.shape != (1, 128):
                    raise ValueError(f"Error on {file}. Embedding must be a (1, 128) numpy array, not {embedding.shape}")

                D, I = faiss_index.search(embedding, 1)
                distance = D[0][0]

                # predicted label must exist in the lookup table.
                predicted_label = source_embeddings[I[0][0]].split("/")[-1].split(".")[0]
                
                if predicted_label not in source_dict:
                    raise ValueError(f"Predicted label {predicted_label} not found in source_dict.")
                
                predicted_source_input_path = source_dict[predicted_label]

                # now get the equivalent reference image.
                reference_label= file.split("/")[-1].split(".")[0]

                if reference_label not in reference_dict:
                    raise ValueError(f"Reference label {reference_label} not found in reference_dict.")

                reference_image_input_path = reference_dict[reference_label]
               
                # to output folder, create a folder based predicted_label.
                output_path = os.path.join(self.output_path, predicted_label)
                os.makedirs(output_path, exist_ok=True)

                # copy everything to the output folder.
                predicted_source_output_path = os.path.join(output_path, os.path.basename(predicted_source_input_path))

                shutil.copy(predicted_source_input_path, predicted_source_output_path)

                reference_image_output_path = os.path.join(output_path, os.path.basename(reference_image_input_path))

                shutil.copy(reference_image_input_path, reference_image_output_path)


        return True


def main():
    return PhotoMatcher()
