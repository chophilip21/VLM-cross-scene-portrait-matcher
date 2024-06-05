"""
run photo matching algorithm
"""

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW, CENTER
import os
from dotenv import load_dotenv
import logging
from photomatcher.enums import IMAGE_EXTENSION

class PhotoMatcher(toga.App):
    """Fronend for the photo matching application."""

    def __init__(self, formal_name=None):
        """Initialize the toga modules."""
        super().__init__(formal_name=formal_name)
        self.home = os.path.expanduser('~')
        env_file = os.path.join(os.path.dirname(__file__), 'resources/config.env')
        load_dotenv(env_file)

    def startup(self):
        """Create the main window for the application."""
        main_box = toga.Box(style=Pack(direction=COLUMN, padding=10, alignment=CENTER))

        # logo
        logo_path = os.path.join(os.path.dirname(__file__), 'resources/logo.jpg')
        if not os.path.isfile(logo_path):
            raise RuntimeError(f"Logo file not found: {logo_path}")
        
        logo = toga.Image(logo_path)
        logo_view = toga.ImageView(logo, style=Pack(width=300, height=300, alignment=CENTER))

        # Source Images Path
        source_path_box = self.create_path_box('Source Images Path:', self.select_source_path)
        self.source_path_input = source_path_box[0]

        # Reference Images Path
        reference_path_box = self.create_path_box('Reference Images Path:', self.select_reference_path)
        self.reference_path_input = reference_path_box[0]

        # Output Path
        output_path_box = self.create_path_box('Output Path:', self.select_output_path)
        self.output_path_input = output_path_box[0]

        # Buttons Box
        buttons_box = toga.Box(style=Pack(direction=ROW, padding=5, alignment=CENTER))
        self.run_button = toga.Button('Run Processing', on_press=self.run_processing, style=Pack(padding=5, background_color='#28a745', color='white', width=200))
        self.refresh_button = toga.Button('Refresh', on_press=self.refresh_inputs, style=Pack(padding=5,background_color='#3545dc', color='white',width=200))
        buttons_box.add(self.run_button)
        buttons_box.add(self.refresh_button)

        # Progress Bar
        self.progress_bar = toga.ProgressBar(max=100, value=0, style=Pack(padding=10, width=1000, alignment=CENTER))

        # Console Log Box
        self.console_log = toga.MultilineTextInput(readonly=True, style=Pack(flex=1, padding=10, background_color='black', color='white', height=150, alignment=CENTER))

        # Adding all components to the main box
        main_box.add(logo_view)
        main_box.add(source_path_box[1])
        main_box.add(reference_path_box[1])
        main_box.add(output_path_box[1])
        main_box.add(buttons_box)
        main_box.add(self.progress_bar)
        main_box.add(self.console_log)

        self.log_message("Welcome to Photo Matcher. Start by selecting the source, reference, and output folders above.")

        self.main_window = toga.MainWindow(title=self.formal_name)
        self.main_window.content = main_box
        self.main_window.show()

    def create_path_box(self, label_text, on_press_handler):
        """Create a box with a label, text input, and button to select a path."""
        path_box = toga.Box(style=Pack(direction=ROW, padding=5, alignment=CENTER))
        path_label = toga.Label(label_text, style=Pack(padding=(0, 5)))
        path_input = toga.TextInput(readonly=True, style=Pack(flex=1))
        path_button = toga.Button('Choose...', on_press=on_press_handler, style=Pack(padding=5))
        path_box.add(path_label)
        path_box.add(path_input)
        path_box.add(path_button)
        return path_input, path_box

    async def select_source_path(self, widget):
        """Select the source images folder."""
        await self.select_path(self.source_path_input, 'Select Source Images Folder')

    async def select_reference_path(self, widget):
        """Select the reference images folder."""
        await self.select_path(self.reference_path_input, 'Select Reference Images Folder')

    async def select_output_path(self, widget):
        """Select the output folder."""
        await self.select_path(self.output_path_input, 'Select Output Folder')

    async def select_path(self, input_widget, dialog_title):
        """Select the input paths using a dialog."""
        try:
            result = await self.main_window.select_folder_dialog(dialog_title, initial_directory=self.home)
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

    async def run_processing(self, widget):
        """Run the photo matching processing."""
        source_path = self.source_path_input.value
        reference_path = self.reference_path_input.value
        output_path = self.output_path_input.value

        if not all([source_path, reference_path, output_path]):
            self.main_window.error_dialog('Error', 'Please select all required folders before running the processing.')
            return

        if len(os.listdir(source_path)) == 0 or len(os.listdir(reference_path)) == 0:
            self.main_window.error_dialog('Error', 'Source and Reference folders must not be empty.')
            return        
        


        self.log_message("Starting processing...")
        await self.run_ml_model(source_path, reference_path, output_path)
        self.main_window.info_dialog('Processing Completed', 'Your photo matching processed successfully!')
        self.log_message("Processing completed.")

    async def run_ml_model(self, source_path: str = None, reference_path: str = None, output_path: str = None):
        """Run ML models here."""
        self.progress_bar.value = 0




        # for i in range(1, 11):
        #     self.progress_bar.value = i * 10
        #     self.log_message(f"Processing... {i * 10}% complete")
        #     await asyncio.sleep(1)  # Simulate processing time

    def refresh_inputs(self, widget):
        """Clear all text inputs and reset progress bar."""
        self.source_path_input.value = ''
        self.reference_path_input.value = ''
        self.output_path_input.value = ''
        self.progress_bar.value = 0
        self.console_log.value = ''
        self.log_message("Inputs refreshed. Start again by selecting the source, reference, and output folders above.")

    def log_message(self, message):
        """Append a message to the console log."""
        self.console_log.value += message + "\n"

def main():
    return PhotoMatcher()
