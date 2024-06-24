"""Main entry point for the application."""
import os
import sys
import signal
from photolink.utils.function import read_config, config_to_env
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
from qss import *

def main():
    """Main entry point for the application."""

    # these will be used to get the path to the temporary directory when the application is bundled
    if getattr(sys, 'frozen', False):
        # Get the path to the temporary directory
        application_path = sys._MEIPASS
    else:
        # Use the script directory for non-bundled execution
        application_path = os.path.dirname(os.path.abspath(__file__))

    config_file = os.path.join(application_path, "./config.ini")
    config = read_config(config_file)
    try:
        config_to_env(config, "MODEL")
    except Exception as e:
        print(f"Error: {e} in reading config file {config_file}. Check again.")
        sys.exit(1)
    os.environ['ROOT_PATH'] = application_path
    app = QApplication(sys.argv)
    
    # import needs to happen after env variable set.
    from app import MainWindow

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