"""Main window for RL pathfinding visualizer."""

import time
from PySide6.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
    QLabel, QSlider, QComboBox, QCheckBox, QSpinBox, QButtonGroup,
    QRadioButton, QStatusBar, QGroupBox, QTextEdit, QFrame, QProgressBar
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut, QColor, QPalette, QCloseEvent

from ..app.controller import RLController
from ..app.fsm import RLState
from .grid_view import GridView


class MainWindow(QMainWindow):
    """Main application window for RL pathfinding."""
    
    def __init__(self, controller: RLController):
        super().__init__()
        self.controller = controller
        
        self.setWindowTitle("RL Pathfinding Visualizer - Q-Learning Agent")
        self.setMinimumSize(1200, 800)
        
        # Statistics tracking
        self.training_start_time = None
        self.current_episode = 0
        self.total_episodes = 0
        self.current_episode_start_time = None  # Track current episode start time
        
        # Timer for updating elapsed time
        self.elapsed_timer = QTimer()
        self.elapsed_timer.timeout.connect(self._update_elapsed_time)
        self.elapsed_timer.timeout.connect(self._update_episode_timing_display)  # Also update episode timing
        self.elapsed_timer.setInterval(1000)  # Update every second
        
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
        content_layout.addWidget(self.grid_view, 3)
        
        # Statistics panel
        stats_panel = self._create_statistics_panel()
        content_layout.addWidget(stats_panel, 1)
        
        main_layout.addLayout(content_layout, 1)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self._update_status_message()
    
    def _create_controls(self) -> QHBoxLayout:
        """Create the control panel."""
        layout = QHBoxLayout()
        
        # RL controls
        rl_group = QGroupBox("Q-Learning")
        rl_layout = QVBoxLayout(rl_group)
        
        # Training mode selection
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Training Mode:"))
        
        self.mode_button_group = QButtonGroup()
        self.background_radio = QRadioButton("Background")
        self.visual_radio = QRadioButton("Visual")
        self.background_radio.setChecked(True)
        
        self.mode_button_group.addButton(self.background_radio)
        self.mode_button_group.addButton(self.visual_radio)
        
        mode_layout.addWidget(self.background_radio)
        mode_layout.addWidget(self.visual_radio)
        mode_layout.addStretch()
        rl_layout.addLayout(mode_layout)
        
        # Speed control for visual mode
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(50, 1000)  # 50ms to 1000ms
        self.speed_slider.setValue(300)  # Default 300ms
        self.speed_slider.setEnabled(False)  # Disabled for background mode initially
        speed_layout.addWidget(self.speed_slider)
        
        self.speed_label = QLabel("300ms")
        self.speed_label.setMinimumWidth(50)
        speed_layout.addWidget(self.speed_label)
        rl_layout.addLayout(speed_layout)
        
        # Action buttons
        buttons_layout = QHBoxLayout()
        self.train_btn = QPushButton("Train")
        self.test_btn = QPushButton("Test")
        self.step_btn = QPushButton("Step")
        self.pause_btn = QPushButton("Pause")
        self.reset_btn = QPushButton("Reset")
        
        self.step_btn.setEnabled(False)  # Initially disabled
        
        for btn in [self.train_btn, self.test_btn, self.step_btn, self.pause_btn, self.reset_btn]:
            buttons_layout.addWidget(btn)
        
        rl_layout.addLayout(buttons_layout)
        
        # Episodes control
        episodes_layout = QVBoxLayout()
        episodes_layout.addWidget(QLabel("Episodes"))
        self.episodes_spin = QSpinBox()
        self.episodes_spin.setRange(10, 15000)  # Extended range for very complex mazes
        self.episodes_spin.setValue(8000)  # Higher default for complex maze learning
        episodes_layout.addWidget(self.episodes_spin)
        
        # Grid controls
        grid_group = QGroupBox("Grid")
        grid_layout = QHBoxLayout(grid_group)
        
        grid_layout.addWidget(QLabel("Size:"))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(7, 50)
        self.width_spin.setValue(15)
        grid_layout.addWidget(self.width_spin)
        
        grid_layout.addWidget(QLabel("×"))
        self.height_spin = QSpinBox()
        self.height_spin.setRange(7, 50)
        self.height_spin.setValue(15)
        grid_layout.addWidget(self.height_spin)
        
        self.new_grid_btn = QPushButton("Empty Grid")
        self.maze_btn = QPushButton("Generate Maze")
        self.save_maze_btn = QPushButton("Save Maze")
        self.load_maze_btn = QPushButton("Load Maze")
        
        grid_layout.addWidget(self.new_grid_btn)
        grid_layout.addWidget(self.maze_btn)
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
        
        # RL parameters
        params_group = QGroupBox("RL Parameters")
        params_layout = QVBoxLayout(params_group)
        
        # Learning rate
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("Learning Rate:"))
        self.lr_spin = QSpinBox()
        self.lr_spin.setRange(1, 50)
        self.lr_spin.setValue(10)  # 0.1
        self.lr_spin.setSuffix(" (×0.01)")
        lr_layout.addWidget(self.lr_spin)
        params_layout.addLayout(lr_layout)
        
        # Epsilon
        eps_layout = QHBoxLayout()
        eps_layout.addWidget(QLabel("Epsilon:"))
        self.eps_spin = QSpinBox()
        self.eps_spin.setRange(1, 50)
        self.eps_spin.setValue(40)  # 0.4 - Even higher exploration for very complex mazes
        self.eps_spin.setSuffix(" (×0.01)")
        eps_layout.addWidget(self.eps_spin)
        params_layout.addLayout(eps_layout)
        
        # Show options
        viz_group = QGroupBox("Visualization")
        viz_layout = QVBoxLayout(viz_group)
        
        self.show_q_values_cb = QCheckBox("Show Q-Values")
        self.show_q_values_cb.setChecked(True)
        viz_layout.addWidget(self.show_q_values_cb)
        
        self.show_costs_cb = QCheckBox("Show Costs")
        viz_layout.addWidget(self.show_costs_cb)
        
        # Smart AI options
        smart_group = QGroupBox("Smart AI")
        smart_layout = QVBoxLayout(smart_group)
        
        self.smart_rewards_cb = QCheckBox("Smart Rewards")
        self.smart_rewards_cb.setChecked(True)
        self.smart_rewards_cb.setToolTip("Enable distance guidance, exploration bonuses, and dead-end penalties")
        smart_layout.addWidget(self.smart_rewards_cb)
        
        self.dead_end_detection_cb = QCheckBox("Dead-End Detection")
        self.dead_end_detection_cb.setChecked(True)
        self.dead_end_detection_cb.setToolTip("Avoid dead ends during exploration and exploitation")
        smart_layout.addWidget(self.dead_end_detection_cb)
        
        self.distance_guidance_cb = QCheckBox("Distance Guidance")
        self.distance_guidance_cb.setChecked(True)
        self.distance_guidance_cb.setToolTip("Bias actions toward goal direction")
        smart_layout.addWidget(self.distance_guidance_cb)
        
        # Testing options
        testing_group = QGroupBox("Testing")
        testing_layout = QVBoxLayout(testing_group)
        
        self.instant_testing_cb = QCheckBox("Instant Testing")
        self.instant_testing_cb.setChecked(True)
        self.instant_testing_cb.setToolTip("Show complete path immediately instead of step-by-step visualization")
        testing_layout.addWidget(self.instant_testing_cb)
        
        # Add all groups to main layout
        layout.addWidget(rl_group)
        layout.addLayout(episodes_layout)
        layout.addWidget(grid_group)
        layout.addWidget(edit_group)
        layout.addWidget(params_group)
        layout.addWidget(viz_group)
        layout.addWidget(smart_group)
        layout.addWidget(testing_group)
        layout.addStretch()
        
        return layout
    
    def _create_statistics_panel(self) -> QGroupBox:
        """Create the statistics display panel."""
        stats_group = QGroupBox("Training Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        # Progress section
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        # Progress info layout
        progress_info_layout = QHBoxLayout()
        
        # Elapsed time
        self.elapsed_time_label = QLabel("Elapsed: 00:00")
        progress_info_layout.addWidget(self.elapsed_time_label)
        
        # Episodes per second
        self.eps_label = QLabel("Rate: 0 eps/s")
        progress_info_layout.addWidget(self.eps_label)
        
        # Estimated time remaining
        self.eta_label = QLabel("ETA: --:--")
        progress_info_layout.addWidget(self.eta_label)
        
        progress_info_layout.addStretch()
        progress_layout.addLayout(progress_info_layout)
        
        # Initially hide the entire progress section
        progress_group.setVisible(False)
        self.progress_group = progress_group  # Store reference
        
        stats_layout.addWidget(progress_group)
        
        # Statistics display
        self.stats_display = QTextEdit()
        self.stats_display.setReadOnly(True)
        self.stats_display.setMaximumHeight(400)
        stats_layout.addWidget(self.stats_display)
        
        # Real-time stats labels
        self.current_state_label = QLabel("State: Idle")
        self.episodes_label = QLabel("Episodes: 0")
        self.success_rate_label = QLabel("Success Rate: 0%")
        self.epsilon_label = QLabel("Epsilon: 0.10")
        
        # Episode timing labels
        self.current_episode_time_label = QLabel("Current Episode: --:--")
        self.best_episode_time_label = QLabel("Best Time: --:--")
        self.worst_episode_time_label = QLabel("Worst Time: --:--")
        self.avg_episode_time_label = QLabel("Avg Time: --:--")
        
        # Training readiness status
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        stats_layout.addWidget(separator)
        
        training_status_group = QGroupBox("Training Status")
        training_status_layout = QVBoxLayout(training_status_group)
        
        # Training readiness indicator
        readiness_layout = QHBoxLayout()
        self.training_status_indicator = QLabel()
        self.training_status_indicator.setMaximumSize(20, 20)
        self.training_status_indicator.setMinimumSize(20, 20)
        self.training_status_indicator.setStyleSheet("background-color: #FF6464; border: 2px solid #000; border-radius: 10px;")
        readiness_layout.addWidget(self.training_status_indicator)
        
        self.training_readiness_label = QLabel("Not Ready for Testing")
        readiness_layout.addWidget(self.training_readiness_label)
        readiness_layout.addStretch()
        training_status_layout.addLayout(readiness_layout)
        
        # Training progress indicators
        self.episodes_progress_label = QLabel("Episodes: 0/50 needed")
        self.success_progress_label = QLabel("Successful: 0/10 needed")
        self.success_rate_progress_label = QLabel("Success Rate: 0%/20% needed")
        
        for label in [self.episodes_progress_label, self.success_progress_label, self.success_rate_progress_label]:
            training_status_layout.addWidget(label)
        
        stats_layout.addWidget(training_status_group)
        
        for label in [self.current_state_label, self.episodes_label, 
                     self.success_rate_label, self.epsilon_label,
                     self.current_episode_time_label, self.best_episode_time_label,
                     self.worst_episode_time_label, self.avg_episode_time_label]:
            stats_layout.addWidget(label)
        
        # Color legend
        legend_group = self._create_color_legend()
        stats_layout.addWidget(legend_group)
        
        return stats_group
    
    def _create_color_legend(self) -> QGroupBox:
        """Create color legend for the grid."""
        legend_group = QGroupBox("Legend")
        legend_layout = QVBoxLayout(legend_group)
        
        colors = [
            ("Start (Mouse)", "#64FF64"),
            ("Target (Cheese)", "#FF6464"),
            ("Wall", "#3C3C3C"),
            ("Current", "#FFFF64"),
            ("Path (Insufficient Training)", "#FFC864"),
            ("Optimal Path (Validated)", "#FFFF64"),  # Yellow for validated training
            ("Training Current", "#FF96FF"),  # Magenta
            ("Training Visited", "#DCB4FF"),  # Light purple
            ("Training Consider", "#FFDC96"),  # Light orange
            ("High Q-Value", "#C8FFC8"),
            ("Low Q-Value", "#FFC8C8")
        ]
        
        for name, color in colors:
            item_layout = QHBoxLayout()
            color_label = QLabel()
            color_label.setStyleSheet(f"background-color: {color}; border: 1px solid black;")
            color_label.setMaximumSize(20, 20)
            text_label = QLabel(name)
            
            item_layout.addWidget(color_label)
            item_layout.addWidget(text_label)
            item_layout.addStretch()
            
            legend_layout.addLayout(item_layout)
        
        return legend_group
    
    def _get_training_status(self) -> dict:
        """Get training status information for UI display."""
        # Call controller's validation method
        is_valid, message = self.controller._validate_training_data()
        
        # Get training statistics
        agent = self.controller.agent
        episodes_completed = agent.episodes_completed
        training_history = agent.training_history
        
        # Calculate current progress
        successful_episodes = sum(1 for ep in training_history if ep.reached_goal) if training_history else 0
        success_rate = successful_episodes / len(training_history) if training_history else 0.0
        
        # Define thresholds (matching controller validation)
        min_episodes = 50
        min_successful = 10
        min_success_rate = 0.2
        
        # Determine status level
        if is_valid:
            status_level = "ready"
            status_color = "#64FF64"  # Green
            status_text = "Ready for Testing"
        elif episodes_completed >= min_episodes / 2 or successful_episodes >= min_successful / 2:
            status_level = "partial"
            status_color = "#FFC864"  # Orange/Yellow
            status_text = "Partially Trained"
        else:
            status_level = "insufficient"
            status_color = "#FF6464"  # Red
            status_text = "Insufficient Training"
        
        return {
            "is_ready": is_valid,
            "message": message,
            "status_level": status_level,
            "status_color": status_color,
            "status_text": status_text,
            "episodes_completed": episodes_completed,
            "episodes_needed": min_episodes,
            "successful_episodes": successful_episodes,
            "successful_needed": min_successful,
            "success_rate": success_rate,
            "success_rate_needed": min_success_rate
        }
    
    def _setup_connections(self):
        """Setup signal connections."""
        # Controller signals
        self.controller.state_changed.connect(self._on_state_changed)
        self.controller.episode_completed.connect(self._on_episode_completed)
        self.controller.training_progress.connect(self._on_training_progress)
        self.controller.training_completed.connect(self._on_training_completed)
        self.controller.testing_completed.connect(self._on_testing_completed)
        self.controller.grid_updated.connect(self._on_grid_updated)
        self.controller.error_occurred.connect(self._on_error_occurred)
        
        # Button connections
        self.train_btn.clicked.connect(self._on_train_clicked)
        self.test_btn.clicked.connect(self._on_test_clicked)
        self.step_btn.clicked.connect(self._on_step_clicked)
        self.pause_btn.clicked.connect(self._on_pause_clicked)
        self.reset_btn.clicked.connect(self._on_reset_clicked)
        
        self.new_grid_btn.clicked.connect(self._on_new_grid_clicked)
        self.maze_btn.clicked.connect(self._on_maze_clicked)
        self.save_maze_btn.clicked.connect(self._on_save_maze)
        self.load_maze_btn.clicked.connect(self._on_load_maze)
        
        # Parameter updates
        self.lr_spin.valueChanged.connect(self._on_params_changed)
        self.eps_spin.valueChanged.connect(self._on_params_changed)
        
        # Visualization toggles
        self.show_q_values_cb.toggled.connect(self.grid_view.set_show_q_values)
        self.show_costs_cb.toggled.connect(self.grid_view.set_show_costs)
        
        # Edit mode
        self.wall_radio.toggled.connect(lambda: self.grid_view.set_edit_mode("wall"))
        self.start_radio.toggled.connect(lambda: self.grid_view.set_edit_mode("start"))
        self.target_radio.toggled.connect(lambda: self.grid_view.set_edit_mode("target"))
        
        # Training mode
        self.background_radio.toggled.connect(self._on_training_mode_changed)
        self.visual_radio.toggled.connect(self._on_training_mode_changed)
        
        # Speed control
        self.speed_slider.valueChanged.connect(self._on_speed_changed)
        
        # Smart AI controls
        self.smart_rewards_cb.toggled.connect(self._on_smart_ai_changed)
        self.dead_end_detection_cb.toggled.connect(self._on_smart_ai_changed)
        self.distance_guidance_cb.toggled.connect(self._on_smart_ai_changed)
        
        # Testing controls
        self.instant_testing_cb.toggled.connect(self._on_testing_options_changed)
    
    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        QShortcut(QKeySequence("Q"), self, self.close)
        QShortcut(QKeySequence("T"), self, self._on_train_clicked)
        QShortcut(QKeySequence("Space"), self, self._on_test_clicked)
        QShortcut(QKeySequence("R"), self, self._on_reset_clicked)
        QShortcut(QKeySequence("Ctrl+S"), self, self._on_save_maze)
        QShortcut(QKeySequence("Ctrl+O"), self, self._on_load_maze)
    
    def _on_state_changed(self, state: RLState):
        """Handle state changes."""
        self._update_button_states()
        self._update_statistics_display()
        
        if state == RLState.TRAINING:
            if self.training_start_time is None:
                self.training_start_time = time.time()
            self.progress_group.setVisible(True)
            self.progress_bar.setMaximum(self.episodes_spin.value())
            self.progress_bar.setValue(0)
            self.elapsed_timer.start()
            self._update_elapsed_time()  # Initial update
            self._update_training_rate()  # Initial rate update
        elif state == RLState.PAUSED:
            # Keep the elapsed timer running during pause so we can see total time
            pass
        else:
            if state in [RLState.IDLE, RLState.CONVERGED]:
                self.progress_group.setVisible(False)
                self.training_start_time = None
            self.elapsed_timer.stop()
    
    def _on_episode_completed(self, episode):
        """Handle episode completion."""
        self.current_episode = episode.number + 1
        self.progress_bar.setValue(self.current_episode)
        self._update_statistics_display()
        self._update_training_rate()
        self._update_episode_timing_display()  # Update timing displays
        
        # Reset episode start time for next episode
        self.current_episode_start_time = time.time()
    
    def _on_training_progress(self, current_episode: int, total_episodes: int):
        """Handle training progress updates."""
        self.current_episode = current_episode
        self.total_episodes = total_episodes
        self.progress_bar.setValue(current_episode)
        if self.progress_bar.maximum() != total_episodes:
            self.progress_bar.setMaximum(total_episodes)
        self._update_training_rate()
        self._update_episode_timing_display()  # Update timing displays
    
    def _on_training_completed(self, result):
        """Handle training completion."""
        self.progress_group.setVisible(False)
        self._update_statistics_display()
        self._update_episode_timing_display()  # Update timing displays
        
        # Calculate elapsed time for completion message
        elapsed_time = ""
        if self.training_start_time is not None:
            elapsed = time.time() - self.training_start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            elapsed_time = f" in {minutes}m {seconds}s"
        
        # Show completion message with statistics
        if result.early_stopped:
            self.status_bar.showMessage(f"Training stopped early{elapsed_time}! {result.stopping_reason} | Success rate: {result.success_rate:.1%} ({result.successful_episodes}/{result.total_episodes})")
        elif result.converged:
            self.status_bar.showMessage(f"Training converged{elapsed_time}! Success rate: {result.success_rate:.1%} ({result.successful_episodes}/{result.total_episodes})")
        else:
            self.status_bar.showMessage(f"Training completed{elapsed_time}. Success rate: {result.success_rate:.1%} ({result.successful_episodes}/{result.total_episodes})")
    
    def _on_testing_completed(self, result):
        """Handle testing completion."""
        if result.success:
            self.status_bar.showMessage(f"Path found! Length: {result.path_length}")
        else:
            self.status_bar.showMessage("No path found - more training may be needed")
    
    def _on_grid_updated(self):
        """Handle grid updates."""
        self.grid_view.fit_in_view()
        self._update_button_states()  # This will also update status message
    
    def _on_error_occurred(self, error_msg: str):
        """Handle errors."""
        self.status_bar.showMessage(f"Error: {error_msg}")
    
    def _on_train_clicked(self):
        """Handle train button click."""
        episodes = self.episodes_spin.value()
        self.total_episodes = episodes
        self.current_episode = 0
        self.training_start_time = time.time()  # Reset start time
        self.current_episode_start_time = time.time()  # Reset episode start time
        self.controller.start_training(episodes)
    
    def _on_test_clicked(self):
        """Handle test button click."""
        self.controller.start_testing()
    
    def _on_step_clicked(self):
        """Handle step button click."""
        if self.controller.current_state == RLState.TRAINING:
            self.controller.step_visual_training()
        elif self.controller.current_state == RLState.TESTING:
            self.controller.step_testing()
    
    def _on_pause_clicked(self):
        """Handle pause button click."""
        if self.controller.current_state == RLState.TRAINING:
            self.controller.pause_training()
        elif self.controller.current_state == RLState.PAUSED:
            self.controller.resume_training()
    
    def _on_reset_clicked(self):
        """Handle reset button click."""
        # Stop any ongoing training first
        if self.controller.current_state in [RLState.TRAINING, RLState.PAUSED]:
            self.controller.stop_training()
        
        self.controller.reset_algorithm()
        self.current_episode = 0
        self.total_episodes = 0
        self.training_start_time = None
        self.current_episode_start_time = None
        self.elapsed_timer.stop()
        self._update_elapsed_time()
        self._update_episode_timing_display()
    
    def _on_new_grid_clicked(self):
        """Handle new grid button click."""
        width = self.width_spin.value()
        height = self.height_spin.value()
        self.controller.create_new_grid(width, height)
        self.controller.place_random_start_target()
    
    def _on_maze_clicked(self):
        """Handle maze generation button click."""
        width = self.width_spin.value()
        height = self.height_spin.value()
        self.controller.generate_maze_grid(width, height)
    
    def _on_save_maze(self):
        """Handle save maze button click."""
        if not self.controller.grid:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Maze", "Please generate or create a maze first.")
            return
        
        # Import maze management functions
        try:
            import sys
            from pathlib import Path
            
            # Add project root to path if not already there
            project_root = Path(__file__).parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            
            from ui.maze_manager import show_save_maze_dialog
            
            # Show save dialog (RL mazes are typically generated, so use "Generated Maze" as default)
            show_save_maze_dialog(
                self.controller.grid,
                self.controller.start_coord,
                self.controller.target_coord,
                "Generated Maze",
                self
            )
        except ImportError as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Import Error", f"Could not import maze manager: {e}")
    
    def _on_load_maze(self):
        """Handle load maze button click."""
        # Set waiting cursor
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        try:
            import sys
            from pathlib import Path
            
            # Add project root to path if not already there
            project_root = Path(__file__).parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            
            from ui.maze_manager import show_maze_manager_dialog
            from utils.maze_serialization import apply_maze_to_grid
            
            # Restore cursor before showing dialog
            QApplication.restoreOverrideCursor()
            
            # Show maze manager dialog
            maze_data = show_maze_manager_dialog(self)
            
            if maze_data:
                # Set waiting cursor for loading operation
                QApplication.setOverrideCursor(Qt.WaitCursor)
                
                print(f"Loading maze: {maze_data.name} ({maze_data.width}x{maze_data.height})")
                
                # Create a new grid with the right dimensions
                from rl.utils.grid_factory import create_empty_grid
                new_grid = create_empty_grid(maze_data.width, maze_data.height)
                
                # Apply the loaded maze to the grid
                apply_maze_to_grid(maze_data, new_grid)
                
                # Stop any ongoing training first
                if self.controller.current_state in [RLState.TRAINING, RLState.PAUSED]:
                    self.controller.stop_training()
                
                # Update controller with new grid and coordinates
                self.controller._grid = new_grid
                self.controller._start_coord = maze_data.start
                self.controller._target_coord = maze_data.target
                
                # Reset algorithm state and Q-values
                self.controller.reset_algorithm()
                
                # Reset all training statistics
                self.current_episode = 0
                self.total_episodes = 0
                self.training_start_time = None
                self.current_episode_start_time = None
                self.elapsed_timer.stop()
                
                # Update UI controls
                self.width_spin.setValue(maze_data.width)
                self.height_spin.setValue(maze_data.height)
                
                # Force refresh of displays
                self._update_elapsed_time()
                self._update_episode_timing_display()
                self._update_statistics_display()
                
                # Emit grid update signal
                self.controller.grid_updated.emit()
                
                # Process events to ensure UI updates
                QApplication.processEvents()
                
                # Restore cursor
                QApplication.restoreOverrideCursor()
                
                # Show success message
                from PySide6.QtWidgets import QMessageBox
                maze_name = maze_data.name or f"Maze {maze_data.width}x{maze_data.height}"
                QMessageBox.information(self, "Maze Loaded", f"Successfully loaded '{maze_name}' for RL training!")
                
                print(f"Maze loaded successfully: {maze_name}")
            else:
                # Restore cursor if no maze was selected
                QApplication.restoreOverrideCursor()
                
        except ImportError as e:
            QApplication.restoreOverrideCursor()
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Import Error", f"Could not import maze manager: {e}")
        except Exception as e:
            QApplication.restoreOverrideCursor()
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Load Error", f"Failed to load maze: {str(e)}")
            import traceback
            print(f"Error loading maze: {e}")
            traceback.print_exc()
    
    def _on_params_changed(self):
        """Handle parameter changes."""
        # Update controller config
        self.controller.update_config(
            learning_rate=self.lr_spin.value() / 100.0,
            epsilon=self.eps_spin.value() / 100.0
        )
    
    def _on_training_mode_changed(self):
        """Handle training mode change, including during active training."""
        if self.visual_radio.isChecked():
            mode = "visual"
            self.speed_slider.setEnabled(True)
        else:
            mode = "background"
            self.speed_slider.setEnabled(False)
        
        # Check if training is currently active
        was_training = self.controller.current_state == RLState.TRAINING
        
        self.controller.update_config(training_mode=mode)
        
        # If training was active, handle the mode transition
        if was_training:
            self.controller.handle_training_mode_transition(mode)
        
        self._update_button_states()
    
    def _on_speed_changed(self, value: int):
        """Handle speed slider change."""
        self.speed_label.setText(f"{value}ms")
        self.controller.update_config(visual_step_delay=value)
    
    def _on_smart_ai_changed(self):
        """Handle smart AI settings change."""
        self.controller.update_config(
            use_smart_rewards=self.smart_rewards_cb.isChecked(),
            detect_dead_ends=self.dead_end_detection_cb.isChecked(),
            use_distance_guidance=self.distance_guidance_cb.isChecked()
        )
    
    def _on_testing_options_changed(self):
        """Handle testing options change."""
        self.controller.update_config(
            instant_testing=self.instant_testing_cb.isChecked()
        )
    
    def _update_elapsed_time(self):
        """Update the elapsed time display."""
        if self.training_start_time is not None:
            elapsed = time.time() - self.training_start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            
            if hours > 0:
                time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            else:
                time_str = f"{minutes:02d}:{seconds:02d}"
            
            self.elapsed_time_label.setText(f"Elapsed: {time_str}")
        else:
            self.elapsed_time_label.setText("Elapsed: 00:00")
    
    def _update_training_rate(self):
        """Update the training rate and ETA display."""
        if self.training_start_time is not None and self.current_episode > 0:
            elapsed = time.time() - self.training_start_time
            if elapsed > 0:
                rate = self.current_episode / elapsed
                if rate >= 10:
                    self.eps_label.setText(f"Rate: {rate:.0f} eps/s")
                elif rate >= 1:
                    self.eps_label.setText(f"Rate: {rate:.1f} eps/s")
                else:
                    self.eps_label.setText(f"Rate: {rate:.2f} eps/s")
                
                # Calculate ETA
                remaining_episodes = self.total_episodes - self.current_episode
                if rate > 0 and remaining_episodes > 0:
                    eta_seconds = remaining_episodes / rate
                    eta_minutes = int(eta_seconds // 60)
                    eta_seconds = int(eta_seconds % 60)
                    
                    if eta_minutes >= 60:
                        eta_hours = int(eta_minutes // 60)
                        eta_minutes = int(eta_minutes % 60)
                        self.eta_label.setText(f"ETA: {eta_hours:02d}:{eta_minutes:02d}:{eta_seconds:02d}")
                    else:
                        self.eta_label.setText(f"ETA: {eta_minutes:02d}:{eta_seconds:02d}")
                else:
                    self.eta_label.setText("ETA: --:--")
            else:
                self.eps_label.setText("Rate: 0 eps/s")
                self.eta_label.setText("ETA: --:--")
        else:
            self.eps_label.setText("Rate: 0 eps/s")
            self.eta_label.setText("ETA: --:--")
    
    def _format_time(self, seconds: float) -> str:
        """Format time in seconds to MM:SS or HH:MM:SS format."""
        if seconds < 0:
            return "--:--"
        
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        
        if minutes >= 60:
            hours = int(minutes // 60)
            minutes = int(minutes % 60)
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    def _update_episode_timing_display(self):
        """Update episode timing displays."""
        agent = self.controller.agent
        
        # Current episode elapsed time
        if self.current_episode_start_time is not None and self.controller.current_state == RLState.TRAINING:
            current_elapsed = time.time() - self.current_episode_start_time
            self.current_episode_time_label.setText(f"Current Episode: {self._format_time(current_elapsed)}")
        else:
            self.current_episode_time_label.setText("Current Episode: --:--")
        
        # Best, worst, and average times from completed episodes
        if agent.training_history:
            successful_times = [ep.elapsed_time for ep in agent.training_history if ep.reached_goal and ep.elapsed_time > 0]
            
            if successful_times:
                best_time = min(successful_times)
                worst_time = max(successful_times)
                avg_time = sum(successful_times) / len(successful_times)
                
                self.best_episode_time_label.setText(f"Best Time: {self._format_time(best_time)}")
                self.worst_episode_time_label.setText(f"Worst Time: {self._format_time(worst_time)}")
                self.avg_episode_time_label.setText(f"Avg Time: {self._format_time(avg_time)}")
            else:
                self.best_episode_time_label.setText("Best Time: --:--")
                self.worst_episode_time_label.setText("Worst Time: --:--")
                self.avg_episode_time_label.setText("Avg Time: --:--")
        else:
            self.best_episode_time_label.setText("Best Time: --:--")
            self.worst_episode_time_label.setText("Worst Time: --:--")
            self.avg_episode_time_label.setText("Avg Time: --:--")
    
    def _update_button_states(self):
        """Update button enabled states based on current state."""
        state = self.controller.current_state
        
        can_start = self.controller.can_start_training()
        is_training = state == RLState.TRAINING
        is_paused = state == RLState.PAUSED
        is_testing = state == RLState.TESTING
        is_visual_mode = self.visual_radio.isChecked()
        
        # Get training status for test button
        training_status = self._get_training_status()
        
        # Update button states
        self.train_btn.setEnabled(can_start and not is_training and not is_testing)
        
        # Test button: enabled only if training is ready and conditions are met
        test_enabled = (state in [RLState.IDLE, RLState.CONVERGED]) and can_start and training_status['is_ready']
        self.test_btn.setEnabled(test_enabled)
        
        # Set tooltip for test button based on training readiness
        if not can_start:
            self.test_btn.setToolTip("Need grid with start and target positions")
        elif not training_status['is_ready']:
            self.test_btn.setToolTip(f"Cannot test yet: {training_status['message']}")
        else:
            self.test_btn.setToolTip("Test the learned policy")
        
        self.step_btn.setEnabled((is_training and is_visual_mode) or (is_testing and self.controller.config.step_mode))
        self.pause_btn.setEnabled(is_training or is_paused)
        self.pause_btn.setText("Resume" if is_paused else "Pause")
        self.reset_btn.setEnabled(not is_testing)
        
        # Update status message
        self._update_status_message()
    
    def _update_status_message(self):
        """Update status bar message based on current state."""
        if not self.controller.grid:
            self.status_bar.showMessage("No grid - Click 'New Grid' or 'Generate Maze'")
        elif not self.controller.start_coord:
            self.status_bar.showMessage("No start position - Click 'Set Start' and click on grid, or click 'New Grid'")
        elif not self.controller.target_coord:
            self.status_bar.showMessage("No target position - Click 'Set Target' and click on grid, or click 'New Grid'")
        elif self.controller.current_state == RLState.TRAINING:
            mode = "Visual" if self.visual_radio.isChecked() else "Background"
            if self.current_episode > 0:
                progress = (self.current_episode / self.total_episodes) * 100
                self.status_bar.showMessage(f"{mode} training in progress... ({progress:.1f}% complete)")
            else:
                self.status_bar.showMessage(f"{mode} training starting...")
        elif self.controller.current_state == RLState.PAUSED:
            self.status_bar.showMessage("Training paused - Click 'Resume' to continue")
        elif self.controller.current_state == RLState.TESTING:
            self.status_bar.showMessage("Testing learned policy...")
        elif self.controller.current_state == RLState.CONVERGED:
            self.status_bar.showMessage("Training converged! Click 'Test' to see the learned path")
        else:
            self.status_bar.showMessage("Ready! Maze loaded with mouse🐭 and cheese🧀 - Click 'Train' to start Q-Learning | Press 'Q' to quit, Ctrl+S to save maze, Ctrl+O to load maze")
    
    def _update_statistics_display(self):
        """Update statistics display."""
        stats = self.controller.get_statistics()
        agent = self.controller.agent
        
        # Update labels
        self.current_state_label.setText(f"State: {stats['state_description']}")
        self.episodes_label.setText(f"Episodes: {stats['episodes_completed']}")
        self.epsilon_label.setText(f"Epsilon: {stats['current_epsilon']:.3f}")
        
        # Update success rate if we have training history
        if agent.training_history:
            recent_episodes = agent.training_history[-100:] if len(agent.training_history) >= 100 else agent.training_history
            successes = sum(1 for ep in recent_episodes if ep.reached_goal)
            success_rate = successes / len(recent_episodes) if recent_episodes else 0
            self.success_rate_label.setText(f"Success Rate: {success_rate:.1%}")
        
        # Update training status information
        training_status = self._get_training_status()
        
        # Update visual indicator
        self.training_status_indicator.setStyleSheet(
            f"background-color: {training_status['status_color']}; border: 2px solid #000; border-radius: 10px;"
        )
        
        # Update readiness label
        self.training_readiness_label.setText(training_status['status_text'])
        
        # Update progress indicators with color coding
        episodes_color = self._get_progress_color(
            training_status['episodes_completed'], 
            training_status['episodes_needed']
        )
        self.episodes_progress_label.setText(
            f"Episodes: {training_status['episodes_completed']}/{training_status['episodes_needed']} needed"
        )
        self.episodes_progress_label.setStyleSheet(f"color: {episodes_color};")
        
        success_color = self._get_progress_color(
            training_status['successful_episodes'], 
            training_status['successful_needed']
        )
        self.success_progress_label.setText(
            f"Successful: {training_status['successful_episodes']}/{training_status['successful_needed']} needed"
        )
        self.success_progress_label.setStyleSheet(f"color: {success_color};")
        
        rate_color = self._get_progress_color(
            training_status['success_rate'], 
            training_status['success_rate_needed'], 
            is_percentage=True
        )
        self.success_rate_progress_label.setText(
            f"Success Rate: {training_status['success_rate']:.1%}/{training_status['success_rate_needed']:.1%} needed"
        )
        self.success_rate_progress_label.setStyleSheet(f"color: {rate_color};")
        
        # Update episode timing
        self._update_episode_timing_display()
        
        # Update detailed stats
        if agent.training_history:
            recent_episodes = agent.training_history[-10:]  # Last 10 episodes
            stats_text = "Recent Episodes:\\n"
            for ep in recent_episodes:
                status = "✓" if ep.reached_goal else "✗"
                time_str = self._format_time(ep.elapsed_time) if ep.elapsed_time > 0 else "--:--"
                stats_text += f"Ep {ep.number}: {status} {ep.steps} steps, R={ep.total_reward:.1f}, T={time_str}\\n"
            
            # Add training status message if not ready
            if not training_status['is_ready']:
                stats_text += f"\\nTraining Status:\\n{training_status['message']}"
            
            self.stats_display.setPlainText(stats_text)
        else:
            self.stats_display.setPlainText("No training data yet.\\n\\nClick 'Train' to start learning before testing.")
    
    def _get_progress_color(self, current: float, needed: float, is_percentage: bool = False) -> str:
        """Get color for progress indicators based on completion level."""
        if is_percentage:
            ratio = current / needed if needed > 0 else 0
        else:
            ratio = current / needed if needed > 0 else 0
        
        if ratio >= 1.0:
            return "#00AA00"  # Dark green - complete
        elif ratio >= 0.75:
            return "#66CC00"  # Light green - nearly complete
        elif ratio >= 0.5:
            return "#FF8800"  # Orange - halfway
        elif ratio >= 0.25:
            return "#FF6600"  # Dark orange - some progress
        else:
            return "#CC0000"  # Dark red - insufficient
    
    def closeEvent(self, event: QCloseEvent):
        """Handle application close event with thorough cleanup to prevent thread contamination."""
        try:
            # Stop the elapsed timer (handle Qt object deletion gracefully)
            try:
                if hasattr(self, 'elapsed_timer') and self.elapsed_timer:
                    self.elapsed_timer.stop()
                    self.elapsed_timer.deleteLater()
            except RuntimeError:
                pass  # Timer already deleted by Qt
            
            # Clean up controller resources (threads, timers, etc.) and wait for completion
            if hasattr(self, 'controller') and self.controller:
                print("Cleaning up controller resources...")
                self.controller.cleanup()
                
                # Additional safety check: ensure training thread is really stopped
                if (hasattr(self.controller, '_training_thread') and 
                    self.controller._training_thread and 
                    self.controller._training_thread.isRunning()):
                    print("Warning: Training thread still running after cleanup, waiting...")
                    # Give it a bit more time to terminate
                    import time
                    for attempt in range(20):  # Up to 200ms total
                        if not self.controller._training_thread.isRunning():
                            break
                        time.sleep(0.01)  # 10ms increments
                        if attempt % 5 == 0:  # Every 50ms
                            print(f"Still waiting for thread termination (attempt {attempt + 1})")
                    
                    # Final check
                    if self.controller._training_thread.isRunning():
                        print("Error: Thread still running, this may cause an abort trap")
                        # Force immediate termination as last resort
                        self.controller._training_thread.terminate()
                        self.controller._training_thread.wait(50)
            
            # Force cleanup of all child widgets and process events multiple times
            try:
                # Get the QApplication instance
                app = self.parent() if self.parent() else None
                if not app:
                    from PySide6.QtWidgets import QApplication
                    app = QApplication.instance()
                
                if app:
                    # Process any pending deleteLater() calls multiple times
                    print("Processing Qt events for cleanup...")
                    for i in range(10):  # More iterations for thorough cleanup
                        app.processEvents()
                        if i == 4:  # Brief pause in the middle
                            import time
                            time.sleep(0.01)
                    
                    # Final event processing after a brief wait
                    import time
                    time.sleep(0.02)  # 20ms wait
                    app.processEvents()
                    print("Qt cleanup completed")
            except Exception as e:
                print(f"Qt cleanup error: {e}")
            
            # Accept the close event
            event.accept()
            
        except Exception as e:
            # Suppress any errors during shutdown and still close
            print(f"Close event error: {e}")
            event.accept()
