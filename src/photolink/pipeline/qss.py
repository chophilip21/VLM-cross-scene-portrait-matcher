"""Put any long CSS styles here to keep the main code clean"""

START_BUTTON_STYLE = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #28a745, stop:1 #218838);
                color: white;
                border: 2px solid black;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #218838, stop:1 #28a745);
            }
        """

REFRESH_BUTTON_STYLE = """QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #007bff, stop:1 #0069d9);
                color: white;
                border: 2px solid black;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0069d9, stop:1 #007bff);
            }"""

BROSWE_BUTTON_STYLE = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 green, stop:1 darkgreen); 
                color: white;
                border: 2px solid black;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 darkgreen, stop:1 green);
            }
            QPushButton:disabled {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 grey, stop:1 darkgrey);
                color: white;
            }
        """

STOP_BUTTON_STYLE = """
    QPushButton {
                background-color: qlineargradient(
                    spread:pad,
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255, 0, 0, 255),
                    stop:1 rgba(139, 0, 0, 255)
                );
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 10px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: qlineargradient(
                    spread:pad,
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255, 50, 50, 255),
                    stop:1 rgba(139, 0, 0, 255)
                );
            }
            QPushButton:pressed {
                background-color: qlineargradient(
                    spread:pad,
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255, 0, 0, 200),
                    stop:1 rgba(139, 0, 0, 200)
                );
            }
"""

NOTIFICATION_STYLE = """
            QMessageBox#customMessageBox {
                background-color: white !important;
                font-family: Arial, Helvetica, sans-serif !important;
                font-size: 14px !important;
            }
            QMessageBox#customMessageBox QLabel {
                color: black !important;
                background-color: white !important;
            }
            QMessageBox#customMessageBox QPushButton {
                background-color: white !important;
                color: #4682B4 !important;  /* Dark Sky Blue */
                border: none !important;
                padding: 5px !important;
            }
            QMessageBox#customMessageBox QPushButton:hover {
                background-color: #e0e0e0 !important;  /* Light grey on hover */
            }
        """

DARK_THEME = """
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
"""

SETTINGS_DESIGN = """
        QDialog {
            background-color: #f0f0f0;
            color: #333333;
            font-family: Arial, sans-serif;
            font-size: 14px;
        }
        QLabel {
            font-size: 14px;
            padding: 5px;
        }
        QPushButton {
            border: none;
            padding: 10px 20px;
            text-align: center;
            font-size: 14px;
            margin: 4px 2px;
        }
        QPushButton#saveButton {
            background-color: #007BFF;
        }
        QPushButton:pressed {
            background-color: #3e8e41;
        }
        QPushButton[objectName="deleteButton"] {
            background-color: grey;
        }
        QComboBox {
            padding: 5px;
            border-radius: 5px;
            border: 1px solid #ccc;
        }
        """
