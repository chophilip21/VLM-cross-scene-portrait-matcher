"""
run photo matching algorithm
"""

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW, CENTER
import os
import asyncio

class PhotoMatcher(toga.App):

    def __init__(self, formal_name=None):
        super().__init__(formal_name=formal_name)

        self.home = os.path.expanduser('~')
    def startup(self):
        main_box = toga.Box(style=Pack(direction=COLUMN, padding=10, alignment=CENTER))

        # logo
        logo_path = os.path.join(os.path.dirname(__file__), 'resources/logo.jpg')

        if not os.path.isfile(logo_path):
            raise RuntimeError(f"Logo file not found: {logo_path}")
        
        logo = toga.Image(logo_path)
        logo_view = toga.ImageView(logo, style=Pack(width=300, height=300, alignment=CENTER))

        # Source Images Path
        source_path_box = toga.Box(style=Pack(direction=ROW, padding=5, alignment=CENTER))
        source_path_label = toga.Label('Source Images Path:', style=Pack(padding=(0, 5)))
        self.source_path_input = toga.TextInput(readonly=True, style=Pack(flex=1))
        source_path_button = toga.Button('Choose...', on_press=self.select_source_path, style=Pack(padding=5))
        source_path_box.add(source_path_label)
        source_path_box.add(self.source_path_input)
        source_path_box.add(source_path_button)

        # Reference Images Path
        reference_path_box = toga.Box(style=Pack(direction=ROW, padding=5, alignment=CENTER))
        reference_path_label = toga.Label('Reference Images Path:', style=Pack(padding=(0, 5)))
        self.reference_path_input = toga.TextInput(readonly=True, style=Pack(flex=1))
        reference_path_button = toga.Button('Choose...', on_press=self.select_reference_path, style=Pack(padding=5))
        reference_path_box.add(reference_path_label)
        reference_path_box.add(self.reference_path_input)
        reference_path_box.add(reference_path_button)

        # Output Path
        output_path_box = toga.Box(style=Pack(direction=ROW, padding=5, alignment=CENTER))
        output_path_label = toga.Label('Output Path:', style=Pack(padding=(0, 5)))
        self.output_path_input = toga.TextInput(readonly=True, style=Pack(flex=1))
        output_path_button = toga.Button('Choose...', on_press=self.select_output_path, style=Pack(padding=5))
        output_path_box.add(output_path_label)
        output_path_box.add(self.output_path_input)
        output_path_box.add(output_path_button)

        # Buttons Box
        buttons_box = toga.Box(style=Pack(direction=ROW, padding=5, alignment=CENTER))
        self.run_button = toga.Button('Run Processing', on_press=self.run_processing, style=Pack(padding=5, width=200))
        self.refresh_button = toga.Button('Refresh', on_press=self.refresh_inputs, style=Pack(padding=5, width=200))
        buttons_box.add(self.run_button)
        buttons_box.add(self.refresh_button)

        # Progress Bar
        self.progress_bar = toga.ProgressBar(max=100, value=0, style=Pack(padding=10, width=1000, alignment=CENTER))

        # Adding all components to the main box
        main_box.add(logo_view)
        main_box.add(source_path_box)
        main_box.add(reference_path_box)
        main_box.add(output_path_box)
        main_box.add(buttons_box)
        main_box.add(self.progress_bar)

        self.main_window = toga.MainWindow(title=self.formal_name)
        self.main_window.content = main_box
        self.main_window.show()

    async def select_source_path(self, widget):
        try:
            result = await self.main_window.select_folder_dialog(
                'Select Source Images Folder', initial_directory=self.home
            )
            if result:
                self.source_path_input.value = result
            else:
                self.source_path_input.value = "No folder selected!"
        except ValueError:
            self.source_path_input.value = "Select folder dialog was canceled"

    async def select_reference_path(self, widget):
        try:
            result = await self.main_window.select_folder_dialog(
                'Select Reference Images Folder', initial_directory=self.home
            )
            if result:
                self.reference_path_input.value = result
            else:
                self.reference_path_input.value = "No folder selected!"
        except ValueError:
            self.reference_path_input.value = "Select folder dialog was canceled"

    async def select_output_path(self, widget):
        try:
            result = await self.main_window.select_folder_dialog(
                'Select Output Folder', initial_directory=self.home
            )
            if result:
                self.output_path_input.value = result
            else:
                self.output_path_input.value = "No folder selected!"
        except ValueError:
            self.output_path_input.value = "Select folder dialog was canceled"

    async def run_processing(self, widget):
        source_path = self.source_path_input.value
        reference_path = self.reference_path_input.value
        output_path = self.output_path_input.value

        if len(source_path) == 0 or len(reference_path) == 0 or len(output_path) == 0:
            self.main_window.error_dialog('Error', 'Please select all required folders before running the processing.')
            return

        # Run ML model
        await self.run_ml_model(source_path, reference_path, output_path)

        # Show processing complete dialog
        self.main_window.info_dialog('Processing Completed', 'Your photo matching processed successfully!')

    async def run_ml_model(self, source_path: str = None, reference_path: str = None, output_path: str = None):
        """Run ML models here."""
        self.progress_bar.value = 0
        for i in range(1, 11):
            # update progress bar
            self.progress_bar.value = i * 10
            await asyncio.sleep(1)  # Simulate processing time

    def refresh_inputs(self, widget):
        """Clear all text inputs and reset progress bar."""
        self.source_path_input.value = ''
        self.reference_path_input.value = ''
        self.output_path_input.value = ''
        self.progress_bar.value = 0

def main():
    return PhotoMatcher()
