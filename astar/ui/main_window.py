"""Main window for A* pathfinding visualizer."""

from PySide6.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
    QLabel, QSlider, QComboBox, QCheckBox, QSpinBox, QButtonGroup,
    QRadioButton, QStatusBar, QGroupBox, QTextEdit, QFrame
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut, QColor, QPalette
import time

from ..app.controller import AStarController
from ..app.fsm import AlgoState
from .grid_view import GridView


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self, controller: AStarController):
        super().__init__()
        self.controller = controller
        
        self.setWindowTitle("A* Pathfinding Visualizer - Mouse & Cheese")
        self.setMinimumSize(1000, 700)  # Increased for statistics panel
        
        # Statistics tracking
        self.algorithm_start_time = None
        self.algorithm_end_time = None
        self.step_count = 0
        
        # Create UI
        self._create_ui()
        self._setup_connections()
        self._setup_shortcuts()
        
        # Initial state
        self._update_button_states()
        self._update_statistics_display()
        
    
    def _create_ui(self):
        """Create the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Controls panel
        controls_layout = self._create_controls()
        main_layout.addLayout(controls_layout)
        
        # Content layout (grid + statistics)
        content_layout = QHBoxLayout()
        
        # Grid view
        self.grid_view = GridView(self.controller)
        content_layout.addWidget(self.grid_view, 3)  # Takes 3/4 of horizontal space
        
        # Statistics panel
        stats_panel = self._create_statistics_panel()
        content_layout.addWidget(stats_panel, 1)  # Takes 1/4 of horizontal space
        
        main_layout.addLayout(content_layout, 1)  # Give content most of the vertical space
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Click to place walls, use controls to run A* | Press 'Q' to quit, Space to step, Enter to run, Ctrl+S to save maze, Ctrl+O to load maze")
    
    def _create_controls(self) -> QHBoxLayout:
        """Create the control panel."""
        layout = QHBoxLayout()
        
        # Algorithm controls
        algo_group = QGroupBox("Algorithm")
        algo_layout = QHBoxLayout(algo_group)
        
        self.step_btn = QPushButton("Step")
        self.run_btn = QPushButton("Run")
        self.pause_btn = QPushButton("Pause")
        self.reset_btn = QPushButton("Reset")
        
        for btn in [self.step_btn, self.run_btn, self.pause_btn, self.reset_btn]:
            algo_layout.addWidget(btn)
        
        # Speed control
        speed_layout = QVBoxLayout()
        speed_layout.addWidget(QLabel("Speed"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(50, 1000)
        self.speed_slider.setValue(300)
        speed_layout.addWidget(self.speed_slider)
        
        # Grid controls
        grid_group = QGroupBox("Grid")
        grid_layout = QHBoxLayout(grid_group)
        
        grid_layout.addWidget(QLabel("Size:"))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(7, 99)
        self.width_spin.setValue(25)
        grid_layout.addWidget(self.width_spin)
        
        grid_layout.addWidget(QLabel("×"))
        self.height_spin = QSpinBox()
        self.height_spin.setRange(7, 99) 
        self.height_spin.setValue(25)
        grid_layout.addWidget(self.height_spin)
        
        self.new_grid_btn = QPushButton("New Grid")
        
        # Maze generation controls
        grid_layout.addWidget(QLabel("Maze:"))
        self.maze_type_combo = QComboBox()
        self.maze_type_combo.addItems(["Classic Maze", "Multi-Path Maze", "Branching Maze"])
        self.maze_type_combo.setCurrentIndex(0)
        self.maze_grid_btn = QPushButton("Generate")
        
        # Maze save/load controls
        self.save_maze_btn = QPushButton("Save Maze")
        self.load_maze_btn = QPushButton("Load Maze")
        
        grid_layout.addWidget(self.new_grid_btn)
        grid_layout.addWidget(self.maze_type_combo)
        grid_layout.addWidget(self.maze_grid_btn)
        grid_layout.addWidget(self.save_maze_btn)
        grid_layout.addWidget(self.load_maze_btn)
        
        # Edit mode
        edit_group = QGroupBox("Edit Mode")
        edit_layout = QVBoxLayout(edit_group)
        
        self.edit_button_group = QButtonGroup()
        self.wall_radio = QRadioButton("Add Walls")
        self.start_radio = QRadioButton("Set Start")
        self.target_radio = QRadioButton("Set Target")
        
        self.wall_radio.setChecked(True)
        
        for radio in [self.wall_radio, self.start_radio, self.target_radio]:
            self.edit_button_group.addButton(radio)
            edit_layout.addWidget(radio)
        
        # Algorithm settings
        algo_settings_group = QGroupBox("Algorithm Settings")
        algo_settings_layout = QVBoxLayout(algo_settings_group)
        
        # Heuristic
        heuristic_layout = QHBoxLayout()
        heuristic_layout.addWidget(QLabel("Heuristic:"))
        self.heuristic_combo = QComboBox()
        self.heuristic_combo.addItems(["manhattan", "euclidean", "diagonal", "octile"])
        self.heuristic_combo.setCurrentText("manhattan")
        heuristic_layout.addWidget(self.heuristic_combo)
        algo_settings_layout.addLayout(heuristic_layout)
        
        # Movement (fixed to 4-direction only)
        movement_layout = QHBoxLayout()
        movement_layout.addWidget(QLabel("Movement:"))
        movement_label = QLabel("4-direction only")
        movement_label.setStyleSheet("font-style: italic; color: #666;")
        movement_layout.addWidget(movement_label)
        algo_settings_layout.addLayout(movement_layout)
        
        # Show costs checkbox
        self.show_costs_cb = QCheckBox("Show Costs")
        algo_settings_layout.addWidget(self.show_costs_cb)
        
        # Add all groups to main layout
        layout.addWidget(algo_group)
        layout.addLayout(speed_layout)
        layout.addWidget(grid_group)
        layout.addWidget(edit_group)
        layout.addWidget(algo_settings_group)
        layout.addStretch()  # Push everything to the left
        
        return layout
    
    def _create_statistics_panel(self) -> QGroupBox:
        """Create the statistics display panel."""
        stats_group = QGroupBox("Algorithm Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        # Create text area for statistics
        self.stats_display = QTextEdit()
        self.stats_display.setReadOnly(True)
        self.stats_display.setMaximumHeight(300)
        self.stats_display.setFont(self.stats_display.font())
        
        stats_layout.addWidget(self.stats_display)
        
        # Real-time stats labels
        self.current_stats_label = QLabel("Current State: Idle")
        self.nodes_explored_label = QLabel("Nodes Explored: 0")
        self.open_set_size_label = QLabel("Open Set Size: 0")
        self.closed_set_size_label = QLabel("Closed Set Size: 0")
        
        for label in [self.current_stats_label, self.nodes_explored_label, 
                     self.open_set_size_label, self.closed_set_size_label]:
            stats_layout.addWidget(label)
        
        # Add color legend
        legend_group = self._create_color_legend()
        stats_layout.addWidget(legend_group)
        
        stats_layout.addStretch()
        return stats_group
    
    def _create_color_legend(self) -> QGroupBox:
        """Create the color legend for pathfinding visualization."""
        legend_group = QGroupBox("Color Legend")
        legend_layout = QVBoxLayout(legend_group)
        
        # Import colors from tiles module
        from .tiles import GridTile
        
        # Define legend items with their colors and descriptions
        legend_items = [
            ("empty", "Empty space (traversable)"),
            ("wall", "Wall (obstacle)"),
            ("start", "Start position (🐭 mouse)"),
            ("target", "Target position (🧀 cheese)"),
            ("open", "Open set (nodes to explore)"),
            ("closed", "Closed set (nodes explored)"),
            ("current", "Current node (being processed)"),
            ("path", "Optimal path (solution)"),
        ]
        
        for state, description in legend_items:
            legend_item = self._create_legend_item(GridTile.COLORS[state], description)
            legend_layout.addWidget(legend_item)
        
        return legend_group
    
    def _create_legend_item(self, color: QColor, description: str) -> QWidget:
        """Create a single legend item with color box and description."""
        item_widget = QWidget()
        item_layout = QHBoxLayout(item_widget)
        item_layout.setContentsMargins(2, 2, 2, 2)
        
        # Color box
        color_box = QFrame()
        color_box.setFixedSize(16, 16)
        color_box.setAutoFillBackground(True)
        
        palette = color_box.palette()
        palette.setColor(QPalette.Window, color)
        color_box.setPalette(palette)
        color_box.setFrameStyle(QFrame.Box | QFrame.Raised)
        
        # Description label
        desc_label = QLabel(description)
        desc_label.setFont(desc_label.font())
        
        item_layout.addWidget(color_box)
        item_layout.addWidget(desc_label)
        item_layout.addStretch()
        
        return item_widget
    
    def _setup_connections(self):
        """Setup signal connections."""
        # Algorithm controls
        self.step_btn.clicked.connect(self._on_step_clicked)
        self.run_btn.clicked.connect(self._on_run_clicked)
        self.pause_btn.clicked.connect(self.controller.pause_algorithm)
        self.reset_btn.clicked.connect(self.controller.reset_algorithm)
        
        # Speed control
        self.speed_slider.valueChanged.connect(self._on_speed_changed)
        
        # Grid controls
        self.new_grid_btn.clicked.connect(self._on_new_grid)
        self.maze_grid_btn.clicked.connect(self._on_maze_grid)
        self.save_maze_btn.clicked.connect(self._on_save_maze)
        self.load_maze_btn.clicked.connect(self._on_load_maze)
        
        # Edit mode
        self.wall_radio.toggled.connect(lambda: self._set_edit_mode("wall"))
        self.start_radio.toggled.connect(lambda: self._set_edit_mode("start"))
        self.target_radio.toggled.connect(lambda: self._set_edit_mode("target"))
        
        # Algorithm settings
        self.heuristic_combo.currentTextChanged.connect(self._on_heuristic_changed)
        self.show_costs_cb.toggled.connect(self.grid_view.set_show_costs)
        
        # Controller signals
        self.controller.state_changed.connect(self._on_state_changed)
        self.controller.error_occurred.connect(self._on_error)
        self.controller.algorithm_completed.connect(self._on_algorithm_completed)
        self.controller.step_completed.connect(self._on_step_completed)
    
    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        # Algorithm controls
        QShortcut(QKeySequence("Space"), self, self._on_step_clicked)
        QShortcut(QKeySequence("Return"), self, self._on_run_clicked)
        QShortcut(QKeySequence("R"), self, self.controller.reset_algorithm)
        
        # Grid controls
        QShortcut(QKeySequence("Ctrl+N"), self, self._on_new_grid)
        QShortcut(QKeySequence("Ctrl+M"), self, self._on_maze_grid)
        QShortcut(QKeySequence("Ctrl+S"), self, self._on_save_maze)
        QShortcut(QKeySequence("Ctrl+O"), self, self._on_load_maze)
        
        # Application controls
        QShortcut(QKeySequence("Q"), self, self.close)
        QShortcut(QKeySequence("Ctrl+Q"), self, self.close)
        QShortcut(QKeySequence("Escape"), self, self.close)
    
    def _on_step_clicked(self):
        """Handle step button click with statistics tracking."""
        if self.algorithm_start_time is None:
            self.algorithm_start_time = time.time()
            self.step_count = 0
        
        self.step_count += 1
        self.controller.step_algorithm()
        self._update_statistics_display()
    
    def _on_run_clicked(self):
        """Handle run/resume button click."""
        if self.controller.current_state == AlgoState.PAUSED:
            self.controller.resume_algorithm()
        else:
            if self.algorithm_start_time is None:
                self.algorithm_start_time = time.time()
                self.step_count = 0
            self.controller.start_algorithm()
    
    def _on_speed_changed(self, value: int):
        """Handle speed slider change."""
        self.controller.speed = value
    
    def _on_new_grid(self):
        """Create a new empty grid."""
        width = self.width_spin.value()
        height = self.height_spin.value()
        self.controller.create_new_grid(width, height)
        self._reset_statistics()
    
    
    def _on_maze_grid(self):
        """Generate the selected type of maze."""
        width = self.width_spin.value()
        height = self.height_spin.value()
        
        # Get selected maze type from dropdown
        selected_maze = self.maze_type_combo.currentText()
        
        if selected_maze == "Classic Maze":
            self.controller.generate_maze_grid(width, height)
            self.status_bar.showMessage("Generated: Classic Maze (single optimal path)")
        elif selected_maze == "Multi-Path Maze":
            self.controller.generate_multipath_maze(width, height)
            self.status_bar.showMessage("Generated: Multi-Path Maze (multiple paths available!)")
        else:  # Branching Maze
            self.controller.generate_branching_maze(width, height)
            self.status_bar.showMessage("Generated: Branching Maze (many decision points!)")
        
        self._reset_statistics()
    
    def _on_save_maze(self):
        """Handle save maze button click."""
        if not self.controller.grid:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Maze", "Please generate or create a maze first.")
            return
        
        # Import maze management functions
        try:
            from ui.maze_manager import show_save_maze_dialog
            
            # Determine generation method based on current selection
            generation_method = self.maze_type_combo.currentText()
            
            # Show save dialog
            show_save_maze_dialog(
                self.controller.grid,
                self.controller.start_coord,
                self.controller.target_coord,
                generation_method,
                self
            )
        except ImportError as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Import Error", f"Could not import maze manager: {e}")
    
    def _on_load_maze(self):
        """Handle load maze button click."""
        try:
            from ui.maze_manager import show_maze_manager_dialog
            from utils.maze_serialization import apply_maze_to_grid
            
            # Show maze manager dialog
            maze_data = show_maze_manager_dialog(self)
            
            if maze_data:
                # Create a new grid with the right dimensions
                from ..domain.grid_factory import create_empty_grid
                new_grid = create_empty_grid(maze_data.width, maze_data.height)
                
                # Apply the loaded maze to the grid
                apply_maze_to_grid(maze_data, new_grid)
                
                # Update controller with new grid and coordinates
                self.controller._grid = new_grid
                self.controller._start_coord = maze_data.start
                self.controller._target_coord = maze_data.target
                
                # Reset algorithm state
                self.controller.reset_algorithm()
                
                # Update UI
                self.width_spin.setValue(maze_data.width)
                self.height_spin.setValue(maze_data.height)
                self.controller.grid_updated.emit()
                
                # Reset statistics
                self._reset_statistics()
                
                # Show success message
                from PySide6.QtWidgets import QMessageBox
                maze_name = maze_data.name or f"Maze {maze_data.width}x{maze_data.height}"
                QMessageBox.information(self, "Maze Loaded", f"Successfully loaded '{maze_name}'!")
                
        except ImportError as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Import Error", f"Could not import maze manager: {e}")
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Load Error", f"Failed to load maze: {str(e)}")
    
    def _set_edit_mode(self, mode: str):
        """Set the grid edit mode."""
        self.grid_view.set_edit_mode(mode)
    
    def _on_heuristic_changed(self, heuristic: str):
        """Handle heuristic change."""
        self.controller.update_config(heuristic=heuristic)
    
    
    def _on_state_changed(self, state: AlgoState):
        """Handle algorithm state change."""
        # Reset statistics when starting fresh
        if state == AlgoState.IDLE:
            self._reset_statistics()
        
        self._update_button_states()
        self._update_statistics_display()
        self.status_bar.showMessage(f"State: {state.value.title()}")
    
    def _on_error(self, error_msg: str):
        """Handle error from controller."""
        self.status_bar.showMessage(f"Error: {error_msg}")
    
    def _on_algorithm_completed(self, result):
        """Handle algorithm completion."""
        # Record completion time
        self.algorithm_end_time = time.time()
        
        if result.success:
            elapsed = self.algorithm_end_time - self.algorithm_start_time if self.algorithm_start_time else 0
            self.status_bar.showMessage(
                f"Path found! Cost: {result.path_cost:.2f}, Nodes explored: {result.nodes_explored}, Time: {elapsed:.2f}s"
            )
        else:
            elapsed = self.algorithm_end_time - self.algorithm_start_time if self.algorithm_start_time else 0
            self.status_bar.showMessage(
                f"No path found. Nodes explored: {result.nodes_explored}, Time: {elapsed:.2f}s"
            )
        
        self._update_statistics_display()
    
    def _update_button_states(self):
        """Update button enabled/disabled states based on current state."""
        state = self.controller.current_state
        
        self.step_btn.setEnabled(state in [AlgoState.IDLE, AlgoState.PAUSED])
        self.run_btn.setText("Resume" if state == AlgoState.PAUSED else "Run")
        self.run_btn.setEnabled(state in [AlgoState.IDLE, AlgoState.PAUSED])
        self.pause_btn.setEnabled(state == AlgoState.RUNNING)
        self.reset_btn.setEnabled(state != AlgoState.IDLE)
    
    def _on_step_completed(self, result):
        """Handle individual step completion."""
        if self.controller.current_state == AlgoState.RUNNING:
            self.step_count += 1
        self._update_statistics_display()
    
    def _reset_statistics(self):
        """Reset all statistics tracking."""
        self.algorithm_start_time = None
        self.algorithm_end_time = None
        self.step_count = 0
        self._update_statistics_display()
    
    def _update_statistics_display(self):
        """Update the statistics display with current data."""
        # Get current statistics from controller
        stats = self.controller.get_statistics()
        
        # Update real-time labels
        self.current_stats_label.setText(f"Current State: {stats['state_description']}")
        self.nodes_explored_label.setText(f"Nodes Explored: {stats['nodes_explored']}")
        self.open_set_size_label.setText(f"Open Set Size: {stats['open_set_size']}")
        self.closed_set_size_label.setText(f"Closed Set Size: {stats['closed_set_size']}")
        
        # Calculate timing information
        current_time = time.time()
        elapsed_time = 0
        
        if self.algorithm_start_time is not None:
            if self.algorithm_end_time is not None:
                elapsed_time = self.algorithm_end_time - self.algorithm_start_time
            else:
                elapsed_time = current_time - self.algorithm_start_time
        
        # Calculate maze density for information
        maze_info = ""
        if self.controller.grid:
            wall_count = sum(1 for node in self.controller.grid.nodes.values() if node.state == "wall")
            total_nodes = len(self.controller.grid.nodes)
            wall_density = (wall_count / total_nodes) * 100
            maze_info = f"\n• Wall Density: {wall_density:.1f}%"
        
        # Create detailed statistics text
        stats_text = f"""=== ALGORITHM STATISTICS ===

Execution Summary:
• Steps Taken: {self.step_count}
• Elapsed Time: {elapsed_time:.2f} seconds
• Steps per Second: {self.step_count / max(elapsed_time, 0.001):.1f}

Search Progress:
• Nodes Explored: {stats['nodes_explored']}
• Open Set Size: {stats['open_set_size']}
• Closed Set Size: {stats['closed_set_size']}
• Algorithm State: {stats['state_description']}

Grid Information:
• Grid Size: {self.controller.grid.width if self.controller.grid else 0}×{self.controller.grid.height if self.controller.grid else 0}
• Start Position: {self.controller.start_coord or 'Not set'}
• Target Position: {self.controller.target_coord or 'Not set'}{maze_info}

Algorithm Configuration:
• Heuristic: {self.controller.config.heuristic}
• Movement: 4-directional only
• Diagonal Movement: Disabled

=== KEYBOARD SHORTCUTS ===
• Space: Step algorithm
• Enter: Run/Resume algorithm
• R: Reset algorithm
• Q/Esc: Quit application
• Ctrl+N: New empty grid
• Ctrl+M: Generate maze
• Ctrl+S: Save current maze
• Ctrl+O: Load saved maze
"""
        
        self.stats_display.setText(stats_text)
    
    def closeEvent(self, event):
        """Handle application close event with thorough cleanup."""
        try:
            # Reset algorithm before closing to clean up any timers
            self.controller.reset_algorithm()
            
            # Ensure all timers and controllers are properly cleaned up
            if hasattr(self.controller, '_timer') and self.controller._timer:
                self.controller._timer.stop()
            
            # Process events to ensure cleanup operations complete
            try:
                from PySide6.QtWidgets import QApplication
                app = QApplication.instance()
                if app:
                    # Process any pending deleteLater() calls multiple times
                    for _ in range(3):
                        app.processEvents()
            except Exception:
                pass
            
            event.accept()
            
        except Exception:
            # Suppress any errors during shutdown and still close
            event.accept()
