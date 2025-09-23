"""Main entry point for RL Pathfinding Visualizer."""

import sys
import os
import signal
import atexit
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
    
    controller = None
    window = None
    
    # Ensure Qt cleanup happens before Python module shutdown
    def on_exit_cleanup():
        try:
            if controller:
                controller.cleanup()
        except Exception:
            pass
        try:
            if app:
                # Force process all pending events multiple times to ensure thread cleanup
                for _ in range(3):
                    app.processEvents()
                app.quit()
        except Exception:
            pass
    
    atexit.register(on_exit_cleanup)
    
    def signal_handler(sig, frame):
        """Handle system signals for graceful shutdown."""
        print(f"\nReceived signal {sig}, shutting down gracefully...")
        try:
            if controller:
                controller.cleanup()
        except Exception:
            pass  # Ignore cleanup errors during signal handling
        try:
            if window:
                window.close()
        except Exception:
            pass  # Ignore close errors during signal handling
        try:
            # Process events to ensure cleanup operations complete
            for _ in range(3):
                app.processEvents()
            app.quit()
        except Exception:
            pass  # Ignore quit errors
        # Force exit if Qt cleanup fails
        sys.exit(0)
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Create controller and main window
        controller = RLController()
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
            if controller:
                controller.cleanup()
        except Exception as cleanup_error:
            print(f"Cleanup error: {cleanup_error}")
        
        # Force Qt application shutdown before Python module cleanup
        try:
            if app:
                # Process any pending events multiple times to ensure thread cleanup
                for _ in range(5):
                    app.processEvents()
                
                # Properly close and delete the application
                app.quit()
                app.deleteLater()
                
                # Force event processing to handle deleteLater() multiple times
                for _ in range(5):
                    app.processEvents()
                    
                # Final safety check - wait briefly for Qt to finish cleanup
                import time
                time.sleep(0.1)
                
        except Exception as qt_cleanup_error:
            print(f"Qt cleanup error: {qt_cleanup_error}")


if __name__ == "__main__":
    sys.exit(main())