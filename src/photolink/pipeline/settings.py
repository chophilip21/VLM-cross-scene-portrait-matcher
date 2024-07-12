import sys
import time
import threading
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QLabel, QPushButton, 
                               QComboBox, QApplication, QHBoxLayout, QFrame)
from PySide6.QtCore import Qt, Signal, QObject
from loguru import logger
from photolink.pipeline.qss import SETTINGS_DESIGN

class Worker(QObject):
    finished = Signal()

    def run(self):
        time.sleep(5)
        self.finished.emit()
        logger.info("Cache deleted")

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super(SettingsDialog, self).__init__(parent)
        self.setWindowTitle("Settings")
        self.setGeometry(300, 200, 400, 300)  # Adjust the height as needed
        self.initUI()
        self.setStyleSheet(self.get_stylesheet())
        self.worker_thread = None

    def initUI(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)  # Adjust spacing between rows

        # Add top title
        title_label = QLabel("Settings Menu", self)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: black; background-color: #f0f0f0; padding: 10px 0;")
        main_layout.addWidget(title_label)

        # Create table frame
        table_frame = QFrame(self)
        table_frame.setFrameShape(QFrame.StyledPanel)
        table_frame.setStyleSheet("background-color: #ffffff; border-radius: 10px; border: 1px solid #dcdcdc;")

        table_layout = QVBoxLayout(table_frame)
        table_layout.setSpacing(10)
        table_layout.setContentsMargins(10, 10, 10, 10)

        # Add table header directly to main_layout
        header_label = QLabel("Cache settings", self)
        header_label.setAlignment(Qt.AlignLeft)
        header_label.setStyleSheet("font-size: 16px; font-weight: bold; color: white; margin: 10px 0; background-color: #8C8CFF;")
        table_layout.addWidget(header_label)

        # Create delete cache section with rounded border
        delete_cache_row = QFrame(self)
        delete_cache_row.setStyleSheet("background-color: #f9f9f9; border-radius: 10px; padding: 10px;")
        delete_cache_layout = QHBoxLayout(delete_cache_row)
        delete_cache_layout.setSpacing(10)  # Adjust spacing within the row
        delete_cache_layout.setContentsMargins(10, 0, 10, 0)
        delete_cache_label = QLabel("Delete cache", self)
        delete_cache_label.setStyleSheet("color: black; font-weight: bold;")
        delete_cache_label.setAlignment(Qt.AlignLeft)

        self.delete_button = QPushButton("Delete", self)
        self.delete_button.setStyleSheet("background-color: grey; color: white; border-radius: 5px;")
        self.delete_button.clicked.connect(self.handle_delete)

        delete_cache_layout.addWidget(delete_cache_label, alignment=Qt.AlignLeft)
        delete_cache_layout.addWidget(self.delete_button, alignment=Qt.AlignRight)
        table_layout.addWidget(delete_cache_row)

        # Create auto delete cache on schedule section with rounded border
        auto_delete_row = QFrame(self)
        auto_delete_row.setStyleSheet("background-color: #f9f9f9; border-radius: 10px; padding: 10px;")
        auto_delete_layout = QHBoxLayout(auto_delete_row)
        auto_delete_layout.setSpacing(10)
        auto_delete_layout.setContentsMargins(10, 0, 10, 0)
        auto_delete_label = QLabel("Auto Delete Cache on Schedule", self)
        auto_delete_label.setStyleSheet("color: black; font-weight: bold;")
        self.combo_box = QComboBox(self)
        self.combo_box.setStyleSheet("background-color: #ffffff; color: #333333; border-radius: 5px;")
        options = ["Delete cache every run", "One day", "One week", "Two weeks", 
                   "One month", "Six months", "One year"]
        self.combo_box.addItems(options)
        self.combo_box.setCurrentIndex(3)

        auto_delete_layout.addWidget(auto_delete_label, alignment=Qt.AlignLeft)
        auto_delete_layout.addWidget(self.combo_box, alignment=Qt.AlignRight)
        table_layout.addWidget(auto_delete_row)

        main_layout.addWidget(table_frame)

        # Add Save button
        self.save_button = QPushButton("Save", self)
        self.save_button.setObjectName("saveButton")
        self.save_button.setStyleSheet("background-color: #007BFF; color: white; border-radius: 5px; padding: 10px 20px;")
        self.save_button.clicked.connect(self.accept)
        save_layout = QHBoxLayout()
        save_layout.addStretch()
        save_layout.addWidget(self.save_button)
        save_layout.addStretch()
        main_layout.addLayout(save_layout)

        self.setLayout(main_layout)

    def get_stylesheet(self):
        return SETTINGS_DESIGN

    def handle_delete(self):
        self.delete_button.setObjectName("deleteButton")
        self.delete_button.setStyleSheet("background-color: red; color: white;")
        self.delete_button.setText("Deleting...")
        self.delete_button.setEnabled(False)

        self.worker = Worker()
        self.worker.finished.connect(self.reset_delete_button)

        self.worker_thread = threading.Thread(target=self.worker.run)
        self.worker_thread.start()

    def reset_delete_button(self):
        self.delete_button.setStyleSheet("background-color: grey; color: white;")
        self.delete_button.setText("Delete")
        self.delete_button.setEnabled(True)

def show_settings():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    dialog = SettingsDialog()
    dialog.exec()

if __name__ == "__main__":
    show_settings()
