from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Create a layout for the main window
        layout = QVBoxLayout()

        # Create a button
        button = QPushButton("Click Me")

        # Add the button to the layout
        layout.addWidget(button)

        # Create a central widget and set the layout
        central_widget = QWidget()
        central_widget.setLayout(layout)

        # Set the central widget of the main window
        self.setCentralWidget(central_widget)

# Create the application and main window
app = QApplication([])
window = MainWindow()

# Show the main window
window.show()

# Start the application event loop
app.exec_()