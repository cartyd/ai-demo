"""Main entry point for A* Pathfinding Visualizer."""

import sys
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt


def main():
    """Main entry point for the application."""
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
    app.setApplicationName("A* Pathfinding Visualizer")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("A* Demo")
    
    # Import UI components (after QApplication is created)
    from .ui.main_window import MainWindow
    from .app.controller import AStarController
    
    controller = None
    window = None
    
    try:
        # Create controller and main window
        controller = AStarController()
        window = MainWindow(controller)
        
        # Show window and run event loop
        window.show()
        return app.exec()
        
    except Exception as e:
        print(f"Application error: {e}")
        return 1
        
    finally:
        # Ensure cleanup even if something goes wrong
        try:
            if controller and hasattr(controller, '_timer'):
                controller._timer.stop()
        except Exception:
            pass
            
        try:
            if app:
                # Process any pending events to ensure cleanup
                for _ in range(3):
                    app.processEvents()
                app.quit()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())