"""In charge of displaying splash screen for model loading."""
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout
from PySide6.QtCore import QThread, Signal
import sys
import time
import requests


# TODO: Move this to server.py
# Define the ServerThread class
class ServerThread(QThread):
    server_ready = Signal(bool)
    
    def run(self):
        # Simulate starting the server with a sleep
        time.sleep(5)  # Replace with actual server start-up code
        # Check server health status
        server_up = self.check_server_health()
        self.server_ready.emit(server_up)
    
    def check_server_health(self):
        try:
            # response = requests.get("http://localhost:8000/healthz")  # Modify URL as needed
            time.sleep(2)  # Simulate server health check
            return True
        except requests.RequestException:
            return False