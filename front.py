"""Divide functional and UI related logic."""

import photolink.utils.enums as enums
from photolink.utils.function import read_config
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QGridLayout,
    QWidget,
    QPushButton,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QSizePolicy,
    QProgressBar,
    QTextEdit,
    QMessageBox,
)
from PySide6.QtGui import QFont
from PySide6.QtSvgWidgets import QSvgWidget
from qss import *
import shutil
from main import get_application_path, get_config_file
from pathlib import Path


class MainWindowFront(QMainWindow):
    def __init__(self):
        """All UI related codes go here."""
        super().__init__()
        self.application_path = get_application_path()
        config = get_config_file(self.application_path)
        self.config = read_config(config)
        self.current_task = enums.Task.SAMPLE_MATCHING.name
        self.cache_dir = self.application_path / Path(".cache")
        print(f"Cache dir: {self.cache_dir}")
        self.setup_cache_dir(self.cache_dir)
        self.drawUI()

    def drawUI(self):
        """Startup by drawing UI elements"""
        # Set default window size
        self.setWindowTitle("PhotoMatcher v.0.01")
        self.setGeometry(100, 100, 800, 600)

        # Create central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Create main layout
        self.main_layout = QVBoxLayout(central_widget)

        # Add a responsive application title
        self.title_label = QLabel("PhotoMatcher v.0.01", self)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.title_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.main_layout.addWidget(self.title_label, alignment=Qt.AlignmentFlag.AlignTop)

        # Create grid layout for the boxes
        self.grid_layout = QGridLayout()

        # Task layout goes here.
        self.matching_color = self.config.get("UI", "MATCHING_TASK_COLOR").split(",")
        self.clustering_color = self.config.get("UI", "CLUSTERING_TASK_COLOR").split(",")
        match_icon = str(self.application_path / Path(self.config.get("IMAGES", "MATCH_ICON")))
        cluster_icon = str(self.application_path / Path(self.config.get("IMAGES", "CLUSTER_ICON")))
        self.sample_match_box = self.create_task_button(
            match_icon, "Sample Match", self.matching_color[0], self.matching_color[1]
        )
        self.cluster_box = self.create_task_button(
            cluster_icon, "Cluster", self.clustering_color[0], self.clustering_color[1]
        )

        # Add boxes to the grid layout
        self.grid_layout.addWidget(self.sample_match_box, 0, 0)
        self.grid_layout.addWidget(self.cluster_box, 0, 1)

        # Add the grid layout to the main layout
        self.main_layout.addLayout(self.grid_layout)

        # Create and add instruction label
        self.instruction_label = QLabel(self)
        self.instruction_label.setStyleSheet("color: white; font-size: 12px;")
        self.instruction_label.setWordWrap(True)
        self.instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(self.instruction_label)

        # Create path selection layout
        self.path_layout = QVBoxLayout()
        self.main_layout.addLayout(self.path_layout)

        # Create and add source and reference path selectors
        self.source_path_selector = self.create_path_selector("Source Path")
        self.reference_path_selector = self.create_path_selector("Reference Path")
        self.output_path_selector = self.create_path_selector("Output Path")

        self.path_layout.addWidget(self.source_path_selector)
        self.path_layout.addWidget(self.reference_path_selector)
        self.path_layout.addWidget(self.output_path_selector)

        # Create processing button layout
        self.processing_layout = QHBoxLayout()
        self.main_layout.addLayout(self.processing_layout)

        self.start_button = QPushButton("Start Processing", self)
        self.start_button.setStyleSheet(START_BUTTON_STYLE)
        self.start_button.setFixedWidth(150)
        self.start_button.clicked.connect(self.process_jobs)
        self.processing_layout.addWidget(self.start_button)

        self.refresh_button = QPushButton("Refresh", self)
        self.refresh_button.setStyleSheet(REFRESH_BUTTON_STYLE)
        self.refresh_button.setFixedWidth(150)
        self.refresh_button.clicked.connect(self.refresh)
        self.processing_layout.addWidget(self.refresh_button)

        # Add progress bar
        self.progress_bar = QProgressBar(self)
        # self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(self.progress_bar)

        # Add console text display
        self.console = QTextEdit(self)
        self.console.setReadOnly(True)
        self.console.setFixedHeight(150)
        self.console.setText(enums.StatusMessage.DEFAULT.value)
        self.main_layout.addWidget(self.console)

        # Set initial selection to "Sample Match"
        self.select_task("Sample Match")

        # Connect the window resize event to a method
        self.resizeEvent = self.on_resize

        # Set initial font for title label
        self.update_font()

    def create_task_button(self, svg_path, button_text, color1=None, color2=None):
        button = QPushButton(self)

        if color1 is not None and color2 is None:
            button.setStyleSheet(f"background-color: {color1}; border: 2px solid black;")

        # use gradient when two colors are provided
        elif color1 is not None and color2 is not None:
            button.setStyleSheet(
                f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {color1}, stop:1 {color2}); border: 2px solid black;"
            )
        else:
            raise ValueError("Color1 must be provided when color2 is provided")

        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        button.setMinimumSize(200, 100)
        button.clicked.connect(self.handle_box_click)

        # Create a layout to hold the SVG widget and text
        layout = QVBoxLayout(button)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Add SVG icon
        svg_widget = QSvgWidget(svg_path, button)
        svg_widget.setStyleSheet("background-color: transparent; border: none;")
        svg_widget.setFixedSize(50, 50)

        # Add the SVG widget and text to the layout
        layout.addWidget(svg_widget, alignment=Qt.AlignmentFlag.AlignHCenter)

        text_label = QLabel(button_text, button)
        text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text_label.setStyleSheet(
            "background-color: transparent; border: none; font-family: Lato Black;"
        )

        # Set font using QFontDatabase
        font = QFont("Arial", 20, QFont.Bold)
        text_label.setFont(font)
        text_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        text_label.update()
        text_label.repaint()
        layout.addWidget(text_label)

        return button

    def create_path_selector(self, label_text):
        container = QWidget(self)
        layout = QHBoxLayout(container)

        label = QLabel(label_text, self)
        layout.addWidget(label)

        line_edit = QLineEdit(self)
        line_edit.setReadOnly(True)
        layout.addWidget(line_edit)

        button = QPushButton("Browse", self)
        button.setStyleSheet(BROSWE_BUTTON_STYLE)
        button.clicked.connect(lambda: self.browse_path(line_edit))
        layout.addWidget(button)

        container.line_edit = line_edit
        container.button = button

        return container

    def browse_path(self, line_edit):
        file_dialog = QFileDialog(self)
        file_dialog.setObjectName("customFileDialog")
        file_dialog.setWindowTitle("Select Directory")
        file_dialog.setStyleSheet("background-color: white; color: black;")
        file_dialog.setFileMode(QFileDialog.Directory)
        file_dialog.setOption(QFileDialog.ShowDirsOnly, False)

        if file_dialog.exec():
            path = file_dialog.selectedFiles()[0]
            if path:
                line_edit.setText(path)

    def display_notification(self, state_enum, message):
        """Display notification message for important messages."""
        message_box = QMessageBox(self)
        message_box.setObjectName("customMessageBox")
        message_box.setWindowTitle(state_enum)
        message_box.setText(message)
        message_box.setStandardButtons(QMessageBox.Ok)
        message_box.setIcon(QMessageBox.Information)
        message_box.setStyleSheet(NOTIFICATION_STYLE)
        message_box.exec()
        self.console.append(f"{state_enum}: {message}")

    def refresh(self):
        """Reset all the tasks."""
        self.source_path_selector.line_edit.setText("")
        self.reference_path_selector.line_edit.setText("")
        self.output_path_selector.line_edit.setText("")
        self.progress_bar.setValue(0)  # Reset progress bar
        self.console.setText(enums.StatusMessage.DEFAULT.value)
        self.current_task = enums.Task.SAMPLE_MATCHING.name
        self.setup_cache_dir(self.cache_dir)

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        self.console.append(f"Progress: {value}%")

    def log_message(self, message: str):
        """ "Log messages to the console."""

        # ignore some messages.
        if "sos" in message.lower():
            return

        self.console.append(message)

    def on_resize(self, event):
        # Adjust font size based on window height
        font_size = max(16, self.height() // 25)
        self.update_font(font_size)
        super().resizeEvent(event)

    def update_font(self, font_size=24):
        font = QFont("Lato Hairline", font_size, QFont.Bold)
        self.title_label.setFont(font)

    def setup_cache_dir(self, cache_dir: Path):
        """clean up cache folders on start up, and recreate dir"""
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    def change_button_status(self, enable=True):
        """Change the status of the start and refresh buttons."""
        self.start_button.setEnabled(enable)
        self.refresh_button.setEnabled(enable)

        if not enable:
            self.start_button.setText("Processing...")
            # make refresh button and start button grey
            self.start_button.setStyleSheet(
                "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 grey, stop:1 darkgrey);"
            )
            self.refresh_button.setStyleSheet(
                "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 grey, stop:1 darkgrey);"
            )

        else:
            self.start_button.setText("Start Processing")
            self.start_button.setStyleSheet(START_BUTTON_STYLE)
            self.refresh_button.setStyleSheet(REFRESH_BUTTON_STYLE)
