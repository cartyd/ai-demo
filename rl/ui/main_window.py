"""Main window for RL pathfinding visualizer."""

from PySide6.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
    QLabel, QSlider, QComboBox, QCheckBox, QSpinBox, QButtonGroup,
    QRadioButton, QStatusBar, QGroupBox, QTextEdit, QFrame, QProgressBar
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut, QColor, QPalette

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
        rl_layout = QHBoxLayout(rl_group)
        
        self.train_btn = QPushButton("Train")
        self.test_btn = QPushButton("Test")
        self.pause_btn = QPushButton("Pause")
        self.reset_btn = QPushButton("Reset")
        
        for btn in [self.train_btn, self.test_btn, self.pause_btn, self.reset_btn]:
            rl_layout.addWidget(btn)
        
        # Episodes control
        episodes_layout = QVBoxLayout()
        episodes_layout.addWidget(QLabel("Episodes"))
        self.episodes_spin = QSpinBox()
        self.episodes_spin.setRange(10, 5000)
        self.episodes_spin.setValue(1000)
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
        
        grid_layout.addWidget(self.new_grid_btn)
        grid_layout.addWidget(self.maze_btn)
        
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
        self.eps_spin.setValue(10)  # 0.1
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
        
        # Add all groups to main layout
        layout.addWidget(rl_group)
        layout.addLayout(episodes_layout)
        layout.addWidget(grid_group)
        layout.addWidget(edit_group)
        layout.addWidget(params_group)
        layout.addWidget(viz_group)
        layout.addStretch()
        
        return layout
    
    def _create_statistics_panel(self) -> QGroupBox:
        """Create the statistics display panel."""
        stats_group = QGroupBox("Training Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        stats_layout.addWidget(self.progress_bar)
        
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
        
        for label in [self.current_state_label, self.episodes_label, 
                     self.success_rate_label, self.epsilon_label]:
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
            ("Path", "#FFC864"),
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
    
    def _setup_connections(self):
        """Setup signal connections."""
        # Controller signals
        self.controller.state_changed.connect(self._on_state_changed)
        self.controller.episode_completed.connect(self._on_episode_completed)
        self.controller.training_completed.connect(self._on_training_completed)
        self.controller.testing_completed.connect(self._on_testing_completed)
        self.controller.grid_updated.connect(self._on_grid_updated)
        self.controller.error_occurred.connect(self._on_error_occurred)
        
        # Button connections
        self.train_btn.clicked.connect(self._on_train_clicked)
        self.test_btn.clicked.connect(self._on_test_clicked)
        self.pause_btn.clicked.connect(self._on_pause_clicked)
        self.reset_btn.clicked.connect(self._on_reset_clicked)
        
        self.new_grid_btn.clicked.connect(self._on_new_grid_clicked)
        self.maze_btn.clicked.connect(self._on_maze_clicked)
        
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
    
    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        QShortcut(QKeySequence("Q"), self, self.close)
        QShortcut(QKeySequence("T"), self, self._on_train_clicked)
        QShortcut(QKeySequence("Space"), self, self._on_test_clicked)
        QShortcut(QKeySequence("R"), self, self._on_reset_clicked)
    
    def _on_state_changed(self, state: RLState):
        """Handle state changes."""
        self._update_button_states()
        self._update_statistics_display()
        
        if state == RLState.TRAINING:
            self.progress_bar.setVisible(True)
            self.progress_bar.setMaximum(self.episodes_spin.value())
            self.progress_bar.setValue(0)
        else:
            self.progress_bar.setVisible(False)
    
    def _on_episode_completed(self, episode):
        """Handle episode completion."""
        self.current_episode = episode.number + 1
        self.progress_bar.setValue(self.current_episode)
        self._update_statistics_display()
    
    def _on_training_completed(self, result):
        """Handle training completion."""
        self.progress_bar.setVisible(False)
        self._update_statistics_display()
        
        # Show completion message
        if result.converged:
            self.status_bar.showMessage(f"Training converged! Success rate: {result.success_rate:.1%}")
        else:
            self.status_bar.showMessage(f"Training completed. Success rate: {result.success_rate:.1%}")
    
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
        self.controller.start_training(episodes)
    
    def _on_test_clicked(self):
        """Handle test button click."""
        self.controller.start_testing()
    
    def _on_pause_clicked(self):
        """Handle pause button click."""
        if self.controller.current_state == RLState.TRAINING:
            self.controller.pause_training()
        elif self.controller.current_state == RLState.PAUSED:
            self.controller.resume_training()
    
    def _on_reset_clicked(self):
        """Handle reset button click."""
        self.controller.reset_algorithm()
        self.current_episode = 0
        self.total_episodes = 0
    
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
    
    def _on_params_changed(self):
        """Handle parameter changes."""
        learning_rate = self.lr_spin.value() / 100.0
        epsilon = self.eps_spin.value() / 100.0
        
        self.controller.update_config(
            learning_rate=learning_rate,
            epsilon=epsilon
        )
    
    def _update_button_states(self):
        """Update button enabled states based on current state."""
        state = self.controller.current_state
        
        can_start = self.controller.can_start_training()
        is_training = state == RLState.TRAINING
        is_paused = state == RLState.PAUSED
        is_testing = state == RLState.TESTING
        
        self.train_btn.setEnabled(can_start and not is_training and not is_testing)
        self.test_btn.setEnabled((state in [RLState.IDLE, RLState.CONVERGED]) and can_start)
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
            self.status_bar.showMessage("Training in progress...")
        elif self.controller.current_state == RLState.TESTING:
            self.status_bar.showMessage("Testing learned policy...")
        elif self.controller.current_state == RLState.CONVERGED:
            self.status_bar.showMessage("Training converged! Click 'Test' to see the learned path")
        else:
            self.status_bar.showMessage("Ready! Maze loaded with mouse🐭 and cheese🧀 - Click 'Train' to start Q-Learning | Press 'Q' to quit")
    
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
        
        # Update detailed stats
        if agent.training_history:
            recent_episodes = agent.training_history[-10:]  # Last 10 episodes
            stats_text = "Recent Episodes:\\n"
            for ep in recent_episodes:
                status = "✓" if ep.reached_goal else "✗"
                stats_text += f"Ep {ep.number}: {status} {ep.steps} steps, R={ep.total_reward:.1f}\\n"
            
            self.stats_display.setPlainText(stats_text)
        else:
            self.stats_display.setPlainText("No training data yet.")