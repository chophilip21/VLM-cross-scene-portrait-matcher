"""Main entry point for the application."""
import os
import sys
import signal
from photolink.utils.function import read_config, config_to_env
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
from app import MainWindow
from qss import *

def main():
    """Main entry point for the application."""
    config_file = os.path.join(os.path.dirname(__file__), "./config.ini")
    config = read_config(config_file)
    config_to_env(config, "MODEL")
    app = QApplication(sys.argv)
    
    # Enable stylesheet propagation
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseStyleSheetPropagationInWidgetStyles, True)

    def signal_handler(sig, frame):
        print("Received shutdown signal:", sig)
        app.quit()

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Set application icon
    icon_path = config['IMAGES']['LOGO_PATH']
    app.setWindowIcon(QIcon(icon_path))

    # Set dark theme with task box selection styles
    app.setStyleSheet(DARK_THEME)
    window = MainWindow()
    window.setWindowIcon(QIcon(icon_path))  # Set window icon
    window.show()

    # # Example usage of the log_message method
    # window.log_message("Application started")

    # Create a timer to periodically check for signals
    timer = QTimer()
    timer.start(500)  # 500 milliseconds

    # Connect the timer to a no-op lambda function
    timer.timeout.connect(lambda: None)

    try:
        # Run the application
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught, exiting...")
        app.quit()

if __name__ == "__main__":
    main()