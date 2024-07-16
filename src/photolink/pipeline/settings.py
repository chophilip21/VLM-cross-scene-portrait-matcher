import sys
import threading
from pathlib import Path

from loguru import logger
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QFrame,
                               QHBoxLayout, QLabel, QPushButton, QVBoxLayout)

from photolink.pipeline.qss import SETTINGS_DESIGN
from photolink.pipeline import get_cache_dir, read_settings, save_dump_settings
from photolink.utils.function import custom_rmtree

class PeriodManager():

    def __init__(self):
        self.delete_period_options = {0: 'Delete cache every run', 1: 'One day', 7: 'One week', 14: 'Two weeks', 30: 'One month', 180: 'Six months', 365: 'One year'}
        self.cache_dir = get_cache_dir()
        self.settings_json =  self.cache_dir/ Path("settings.json")
        self.settings_dict = read_settings(self.settings_json)
        self.save_period = self.settings_dict["save_period"]

class SettingSignals(QObject):
    cache_deleted = Signal()
    saved = Signal()

# instantiate the objects
pm = PeriodManager()
signals_object = SettingSignals()

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super(SettingsDialog, self).__init__(parent)
        self.setWindowTitle("Settings")
        self.setGeometry(300, 200, 400, 300)  # Adjust the height as needed
        self.initUI()
        self.setStyleSheet(self.get_stylesheet())
        self.worker_thread = None
        self.worker_signal = signals_object

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
        delete_cache_label = QLabel("Delete cache Manually", self)
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
        options = [value for value in pm.delete_period_options.values()]
        self.combo_box.addItems(options)

        # set the default index based on the self.save_period
        current_index = list(pm.delete_period_options.keys()).index(pm.save_period)
        self.combo_box.setCurrentIndex(current_index)

        auto_delete_layout.addWidget(auto_delete_label, alignment=Qt.AlignLeft)
        auto_delete_layout.addWidget(self.combo_box, alignment=Qt.AlignRight)
        table_layout.addWidget(auto_delete_row)

        main_layout.addWidget(table_frame)

        self.save_button = QPushButton("Save", self)
        self.save_button.setObjectName("saveButton")
        self.save_button.setStyleSheet("background-color: #007BFF; color: white; border-radius: 5px; padding: 10px 20px;")
        self.save_button.clicked.connect(self.save_settings)

        save_layout = QHBoxLayout()
        save_layout.addStretch()
        save_layout.addWidget(self.save_button)
        save_layout.addStretch()
        main_layout.addLayout(save_layout)

        self.setLayout(main_layout)

    def save_settings(self):
        """Accept the dialog and save the settings."""
        # get the selected index from the combo box, and update. 
        selected_index = self.combo_box.currentIndex()
        pm.settings_dict["save_period"] = list(pm.delete_period_options.keys())[selected_index]
        pm.save_period = pm.settings_dict["save_period"]

        # save the settings to the settings.json file
        save_dump_settings(pm.settings_json, pm.settings_dict)
        self.accept()
        self.worker_signal.saved.emit()
        logger.info("Settings saved")

    def get_stylesheet(self):
        return SETTINGS_DESIGN

    def delete_cache_immediately(self):
        """Immediately flush out the cache and send signals."""
        custom_rmtree(pm.cache_dir)
        logger.info("Cache deleted")
        self.worker_signal.cache_deleted.emit()

    def handle_delete(self):
        """When deleting cache, it must be handled in a separate thread."""
        self.delete_button.setObjectName("deleteButton")
        self.delete_button.setStyleSheet("background-color: red; color: white;")
        self.delete_button.setText("Deleting...")
        self.delete_button.setEnabled(False)
        self.worker_signal.cache_deleted.connect(self.reset_delete_button)

        # must be deleted on 
        self.worker_thread = threading.Thread(target=self.delete_cache_immediately)
        self.worker_thread.start()

    def reset_delete_button(self):
        self.delete_button.setStyleSheet("background-color: grey; color: white;")
        self.delete_button.setText("Delete")
        self.delete_button.setEnabled(True)

def show_settings():
    """Display settings related content."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    try:
        dialog = SettingsDialog()
        dialog.exec()
    finally:
        # try doing some clean up in case. 
        if dialog.worker_thread is not None:
            logger.info("Cleaning up worker thread for settings dialog.")
            dialog.worker_thread.join()

if __name__ == "__main__":
    show_settings()
