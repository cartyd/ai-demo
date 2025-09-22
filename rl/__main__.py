"""Main entry point for RL Pathfinding Visualizer."""

import sys
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt


def main():
    """Main entry point for the RL pathfinding application."""
    # Set comprehensive environment variables to fix DPI scaling issues on macOS
    os.environ.setdefault('QT_AUTO_SCREEN_SCALE_FACTOR', '0')
    os.environ.setdefault('QT_SCALE_FACTOR', '1')
    os.environ.setdefault('QT_SCREEN_SCALE_FACTORS', '1')
    os.environ.setdefault('QT_DEVICE_PIXEL_RATIO', '1')
    os.environ.setdefault('QT_LOGGING_RULES', 'qt.qpa.backingstore=false')
    
    # Set high DPI policy before creating QApplication to disable scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.Floor)
    
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("RL Pathfinding Visualizer")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("RL Demo")
    
    # Import UI components (after QApplication is created)
    from .ui.main_window import MainWindow
    from .app.controller import RLController
    
    # Create controller and main window
    controller = RLController()
    window = MainWindow(controller)
    
    # Show window and run event loop
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())