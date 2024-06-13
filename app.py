"""main application"""
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, 
    QComboBox, QPushButton, QTextEdit, QFileDialog, QLineEdit, QStackedWidget
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
import os

class MainWindow(QMainWindow):
    """Frontend application main window."""
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("My PySide6 Application")
        self.resize(800, 600)  # Set default window size
        
        # Load the logo image
        self.logo_label = QLabel(self)
        self.logo_path = os.getenv("LOGO_PATH")
        pixmap = QPixmap(self.logo_path)
        pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)  # Resize the logo
        self.logo_label.setPixmap(pixmap)
        self.logo_label.setAlignment(Qt.AlignCenter)
        
        # Create a drop-down menu
        self.task_type_combo = QComboBox(self)
        self.task_type_combo.addItem("Sample Matching")
        self.task_type_combo.addItem("Clustering")
        self.task_type_combo.currentIndexChanged.connect(self.update_task_type)
        
        # Create source directory selection fields for Sample Matching
        self.sample_source_dir_edit = QLineEdit(self)
        self.sample_source_dir_edit.setPlaceholderText("Select path")
        self.sample_source_dir_button = QPushButton("Select Source")
        self.sample_source_dir_button.setObjectName("sample_source_dir_button")
        self.sample_source_dir_button.clicked.connect(self.select_sample_source_directory)
        
        sample_source_layout = QHBoxLayout()
        sample_source_layout.addWidget(self.sample_source_dir_edit)
        sample_source_layout.addWidget(self.sample_source_dir_button)
        
        # Create reference directory selection fields for Sample Matching
        self.sample_reference_dir_edit = QLineEdit(self)
        self.sample_reference_dir_edit.setPlaceholderText("Select path")
        self.sample_reference_dir_button = QPushButton("Select Reference")
        self.sample_reference_dir_button.setObjectName("sample_reference_dir_button")
        self.sample_reference_dir_button.clicked.connect(self.select_sample_reference_directory)
        
        sample_reference_layout = QHBoxLayout()
        sample_reference_layout.addWidget(self.sample_reference_dir_edit)
        sample_reference_layout.addWidget(self.sample_reference_dir_button)
        
        # Create sample matching layout
        sample_matching_layout = QVBoxLayout()
        sample_matching_layout.addLayout(sample_source_layout)
        sample_matching_layout.addLayout(sample_reference_layout)
        
        sample_matching_widget = QWidget()
        sample_matching_widget.setLayout(sample_matching_layout)
        
        # Create source directory selection fields for Clustering
        self.cluster_source_dir_edit = QLineEdit(self)
        self.cluster_source_dir_edit.setPlaceholderText("Select path")
        self.cluster_source_dir_button = QPushButton("Select Source")
        self.cluster_source_dir_button.setObjectName("cluster_source_dir_button")
        self.cluster_source_dir_button.clicked.connect(self.select_cluster_source_directory)
        
        cluster_source_layout = QHBoxLayout()
        cluster_source_layout.addWidget(self.cluster_source_dir_edit)
        cluster_source_layout.addWidget(self.cluster_source_dir_button)
        
        # Create clustering layout
        clustering_layout = QVBoxLayout()
        clustering_layout.addLayout(cluster_source_layout)
        
        clustering_widget = QWidget()
        clustering_widget.setLayout(clustering_layout)
        
        # Stacked widget to switch between task-specific fields
        self.task_stacked_widget = QStackedWidget(self)
        self.task_stacked_widget.addWidget(sample_matching_widget)
        self.task_stacked_widget.addWidget(clustering_widget)
        
        # Create buttons
        self.process_button = QPushButton("Process")
        self.process_button.setObjectName("process_button")
        self.process_button.clicked.connect(self.process_task)
        
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setObjectName("refresh_button")
        self.refresh_button.clicked.connect(self.refresh_task)
        
        # Create a horizontal layout for the buttons and align them to the center
        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        button_layout.addWidget(self.process_button)
        button_layout.addWidget(self.refresh_button)
        button_layout.addStretch(1)
        
        # Create a console log box
        self.console_log = QTextEdit(self)
        self.console_log.setReadOnly(True)  # Make it read-only
        
        # Create a vertical layout and add the widgets to it
        layout = QVBoxLayout()
        layout.addWidget(self.logo_label)
        layout.addWidget(self.task_type_combo)
        layout.addWidget(self.task_stacked_widget)
        layout.addLayout(button_layout)
        layout.addWidget(self.console_log)
        
        # Set the layout to a central widget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
    
    def select_sample_source_directory(self):
        """Open a file dialog to select a source directory for Sample Matching."""
        directory = QFileDialog.getExistingDirectory(self, "Select Sample Source Directory")
        if directory:
            self.sample_source_dir_edit.setText(directory)
    
    def select_sample_reference_directory(self):
        """Open a file dialog to select a reference directory for Sample Matching."""
        directory = QFileDialog.getExistingDirectory(self, "Select Sample Reference Directory")
        if directory:
            self.sample_reference_dir_edit.setText(directory)
    
    def select_cluster_source_directory(self):
        """Open a file dialog to select a source directory for Clustering."""
        directory = QFileDialog.getExistingDirectory(self, "Select Cluster Source Directory")
        if directory:
            self.cluster_source_dir_edit.setText(directory)
    
    def update_task_type(self, index):
        """Update the visible input fields based on the selected task type."""
        self.task_stacked_widget.setCurrentIndex(index)
    
    def process_task(self):
        """Handle the process button click event."""
        self.process_button.setText("Processing")
        # Add your processing code here
        self.log_message("Processing started...")
        # Simulate processing
        QApplication.processEvents()
        # Once done, reset the button text
        self.process_button.setText("Process")
    
    def refresh_task(self):
        """Handle the refresh button click event."""
        self.sample_source_dir_edit.setText("")
        self.sample_reference_dir_edit.setText("")
        self.cluster_source_dir_edit.setText("")
        self.log_message("Paths cleared and reset to default")
    
    def log_message(self, message):
        """Log a message to the console log box."""
        self.console_log.append(message)