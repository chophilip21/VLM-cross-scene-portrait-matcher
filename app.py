from PySide6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QGridLayout, QWidget, QPushButton, QSizePolicy)
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont, QFontDatabase

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set default window size
        self.setWindowTitle("PhotoMatcher v.0.01")
        self.setGeometry(100, 100, 800, 600)

        # Create central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Create main layout
        main_layout = QVBoxLayout(central_widget)

        # Add a responsive application title
        self.title_label = QLabel("PhotoMatcher v.0.01", self)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.title_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        main_layout.addWidget(self.title_label, alignment=Qt.AlignmentFlag.AlignTop)
    
        # Create grid layout for the boxes
        grid_layout = QGridLayout()

        # Create sample match with two blue gradient colors
        self.matching_color = ["#830ec2", "#4a2cf0"]
        self.clustering_color = ["#fbb13c", "#f98301"]
        self.sample_match_box = self.create_task_button("assets/img/match.svg", "Sample Match", self.matching_color[0], self.matching_color[1])
        self.cluster_box = self.create_task_button("assets/img/cluster.svg", "Cluster", self.clustering_color[0], self.clustering_color[1])

        # Add boxes to the grid layout
        grid_layout.addWidget(self.sample_match_box, 0, 0)
        grid_layout.addWidget(self.cluster_box, 0, 1)

        # Add the grid layout to the main layout
        main_layout.addLayout(grid_layout)

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
            button.setStyleSheet(f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {color1}, stop:1 {color2}); border: 2px solid black;")
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
        text_label.setStyleSheet("background-color: transparent; border: none;")

        # Set font using QFontDatabase
        available_fonts = QFontDatabase.families()
        # print(available_fonts)
        font = QFont("Arial", 20, QFont.Bold, italic=True) 
        text_label.setFont(font)
        text_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        text_label.update()
        text_label.repaint()
        # print('wtf is this?', text_label.fontInfo().family())
        layout.addWidget(text_label)

        return button

    @Slot()
    def handle_box_click(self):
        # Reset border colors for both boxes
        self.sample_match_box.setStyleSheet(f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {self.matching_color[0]}, stop:1 {self.matching_color[1]}); border: 2px solid black;")
        self.cluster_box.setStyleSheet(f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {self.clustering_color[0]}, stop:1 {self.clustering_color[1]}); border: 2px solid black;")

        # Highlight the clicked box
        clicked_button = self.sender()
        clicked_button.setStyleSheet(clicked_button.styleSheet() + " border: 2px solid white;")

    def on_resize(self, event):
        # Adjust font size based on window height
        font_size = max(16, self.height() // 25)
        self.update_font(font_size)
        super().resizeEvent(event)

    def update_font(self, font_size=24):
        font = QFont("Lato Hairline", font_size, QFont.Bold)
        self.title_label.setFont(font)
