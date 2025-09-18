"""Main entry point for A* Pathfinding Visualizer."""

import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt


def main():
    """Main entry point for the application."""
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("A* Pathfinding Visualizer")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("A* Demo")
    
    # Enable high DPI support
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # Import UI components (after QApplication is created)
    from .ui.main_window import MainWindow
    from .app.controller import AStarController
    
    # Create controller and main window
    controller = AStarController()
    window = MainWindow(controller)
    
    # Show window and run event loop
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())