"""
run photo matching algorithm
"""
import os
from dotenv import load_dotenv
env_file = os.path.join(os.path.dirname(__file__), "resources/config.env")
load_dotenv(env_file)

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW, CENTER
import os
import logging
from photomatcher.enums import IMAGE_EXTENSION
from concurrent.futures import ProcessPoolExecutor, as_completed
import photomatcher.model.yunet as yunet
import photomatcher.model.sface as sface
import faiss
import shutil


class PhotoMatcher(toga.App):
    """Fronend for the photo matching application."""

    def __init__(self, formal_name=None):
        """Initialize the toga modules."""
        super().__init__(formal_name=formal_name)
        self.home = os.path.expanduser("~")
        self.num_processes = os.cpu_count()
        self.faiss_index = faiss.IndexFlatL2(128)
  
    def startup(self):
        """Create the main window for the application."""
        main_box = toga.Box(style=Pack(direction=COLUMN, padding=10, alignment=CENTER))

        # logo
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
        await self.select_path(
            self.reference_path_input, "Reference Images Folder"
        )

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
        self.fail_path = self.output_path_input.value + "/missed"
        os.makedirs(self.fail_path, exist_ok=True)

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

        self.log_message("Starting processing...")
        result = await self._run_processing()

        # if none returned, do not proceed.
        if result is None:
            self.log_message("Processing failed. Check the inputs again.")
            return

        self.main_window.info_dialog(
            "Processing Completed", "Your photo matching processed successfully!"
        )
        self.log_message("Processing completed.")

    async def _run_processing(
        self,
    ):
        """Run ML models here."""
        self.progress_bar.start()
        self.progress_bar.value = 0

        source_list_images = [
            os.path.join(self.source_path, file)
            for file in os.listdir(self.source_path)
            if file.split('.')[-1] in IMAGE_EXTENSION
        ]

        if len(source_list_images) == 0:
            self.main_window.error_dialog(
                "Error",
                "Please make sure that there are image files in the source folder.",
            )
            return

        reference_list_images = [
            os.path.join(self.reference_path, file)
            for file in os.listdir(self.reference_path)
            if file.split('.')[-1] in IMAGE_EXTENSION
        ]

        if len(reference_list_images) == 0:
            self.main_window.error_dialog(
                "Error",
                "Please make sure that there are image files in the reference folder.",
            )
            return
        

        with ProcessPoolExecutor(max_workers=self.num_processes) as cpu_executor:
        
            # submit jobs for sources
            result_source = [
                cpu_executor.submit(self._run_ml_model, source_file)
                for source_file in source_list_images
            ]

            for source_jobs in as_completed(result_source):
                result = source_jobs.result()

                # add it to the faiss index
                if len(result) > 0:
                    for embedding in result:
                        self.faiss_index.add(embedding)

                # update progress for 50% of the total progress
                self.progress_bar.value = self.progress_bar.value + (50 / len(source_list_images))


            #submit jobs for references
            result_reference = [
                cpu_executor.submit(self._run_ml_model, reference_file)
                for reference_file in reference_list_images
            ]

            for reference_jobs in as_completed(result_reference):
                result = reference_jobs.result()

                # search the faiss index
                if len(result) > 0:
                    for embedding in result:
                        D, I = self.faiss_index.search(embedding, 1)

                        print(D, I)

                # update progress for 50% of the total progress
                self.progress_bar.value = self.progress_bar.value + (50 / len(reference_list_images))

        self.progress_bar.stop()

    def _run_ml_model(self, image_path: str):
        """Process face detection and recognition for images"""
        detection_result = yunet.run_face_detection(image_path)

        failed_image = os.path.join(self.fail_path, os.path.basename(image_path))

        if "error" in detection_result:
            self.log_message(
                f"Face detection error on source image {image_path}: {detection_result['error']}"
            )
            shutil.copy(image_path, failed_image)
            return

        image = detection_result["image"]
        embedding_dict = sface.run_face_recognition(image, detection_result["faces"])

        if "error" in embedding_dict:
            self.log_message(
                f"Face recognition error on source image {image_path}: {embedding_dict['error']}"
            )
            shutil.copy(image_path, failed_image)
            return
        
        embedding = embedding_dict["embeddings"]

        return embedding


def main():
    return PhotoMatcher()
