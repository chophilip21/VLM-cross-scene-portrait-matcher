import os
import sys
import signal
from dotenv import load_dotenv
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
from app import MainWindow

def main():
    """Main entry point for the application."""
    env_file = os.path.join(os.path.dirname(__file__), "./config.env")
    load_dotenv(dotenv_path=env_file)

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
    icon_path = os.getenv("LOGO_PATH")
    app.setWindowIcon(QIcon(icon_path))

    # Set dark theme with task box selection styles
    dark_theme = """
        QWidget {
            background-color: #2e2e2e;
            color: #f0f0f0;
            font-family: Arial, Helvetica, sans-serif;
        }
        QLineEdit, QComboBox, QTextEdit {
            background-color: #3e3e3e;
            border: 1px solid #5a5a5a;
            padding: 5px;
        }
        QPushButton {
            border: 2px solid transparent;
            background-color: transparent;
            color: inherit;
            padding: 5px 10px;
            font-family: Arial, Helvetica, sans-serif;
            font-size: 14px;
            border-radius: 5px;
        }
        QPushButton:hover {
            background-color: #5a5a5a;
        }
        QPushButton#process_button {
            border-color: #00CC00;  /* Lighter green */
            color: #00CC00;
        }
        QPushButton#refresh_button {
            border-color: #87CEEB;  /* Sky blue */
            color: #87CEEB;
        }
        QPushButton#sample_source_dir_button, QPushButton#sample_reference_dir_button, QPushButton#cluster_source_dir_button {
            border-color: #FFA500;
            color: #FFA500;
        }
        QLabel {
            color: #f0f0f0;
        }
        QWidget#sample_matching, QWidget#clustering {
            background-color: #2596be;
            border: 2px solid transparent;
        }
        QWidget#clustering {
            background-color: #ac2cf0;
        }
        QWidget#task-box-selected {
            border: 2px solid white;
        }
    """
    app.setStyleSheet(dark_theme)
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