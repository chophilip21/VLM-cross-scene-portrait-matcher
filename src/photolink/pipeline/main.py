"""Main entry point for the application."""

import signal
import sys

from loguru import logger
from PySide6.QtCore import QCoreApplication, QProcess, Qt, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from photolink import get_application_path, get_config
from photolink.pipeline.qss import *
from photolink.utils.function import config_to_env


def restart_application():
    logger.warning("Hard restarting application...")
    QCoreApplication.quit()
    status = QProcess.startDetached(sys.executable, sys.argv)
    sys.exit(status)


def run():
    """Main entry point for the application."""

    application_path = get_application_path()
    config_data = get_config()

    # Init global logger
    logger_path = application_path / "worker.log"
    logger.add(
        logger_path, format="{time}:{level}:{message}", level="INFO", enqueue=True
    )

    try:
        config_to_env(config_data, "MODEL")
    except Exception as e:
        logger.error(f"Error: {e} in reading config file to env variable. Check again.")
        sys.exit(1)

    app = QApplication(sys.argv)

    # import needs to happen after env variable set.
    from photolink.pipeline.app import MainWindow

    # Enable stylesheet propagation
    QApplication.setAttribute(
        Qt.ApplicationAttribute.AA_UseStyleSheetPropagationInWidgetStyles, True
    )

    def signal_handler(sig, frame):
        logger.warning("Received shutdown signal:", sig)
        app.quit()

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Set application icon
    icon_path = config_data["IMAGES"]["LOGO_PATH"]
    app.setWindowIcon(QIcon(icon_path))

    # Set dark theme with task box selection styles
    app.setStyleSheet(DARK_THEME)
    window = MainWindow()
    window.setWindowIcon(QIcon(icon_path))  # Set window icon

    # allow hard reset.
    window.refresh_requested.connect(restart_application)
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
        logger.warning("KeyboardInterrupt caught, exiting...")
        app.quit()


if __name__ == "__main__":
    run()
