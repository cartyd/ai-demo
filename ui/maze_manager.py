"""
Maze manager dialog for saving and loading mazes.
Can be used by both A* and RL applications.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QTextEdit,
    QPushButton, QListWidget, QListWidgetItem, QMessageBox, QComboBox,
    QGroupBox, QSplitter, QWidget, QFormLayout, QDialogButtonBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from ..utils.maze_serialization import (
    MazeData, save_maze, load_maze, list_saved_mazes, delete_maze,
    generate_maze_filename, extract_maze_from_grid
)


class SaveMazeDialog(QDialog):
    """Dialog for saving a maze with metadata."""
    
    def __init__(self, maze_data: MazeData, parent=None):
        super().__init__(parent)
        self.maze_data = maze_data
        self.setWindowTitle("Save Maze")
        self.setModal(True)
        self.resize(400, 300)
        
        self._setup_ui()
        self._populate_fields()
    
    def _setup_ui(self):
        """Setup the UI elements."""
        layout = QVBoxLayout(self)
        
        # Form for maze metadata
        form_layout = QFormLayout()
        
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Enter a name for this maze")
        form_layout.addRow("Name:", self.name_edit)
        
        self.generation_combo = QComboBox()
        self.generation_combo.addItems([
            "Classic Maze", "Multi-Path Maze", "Branching Maze", 
            "Random Maze", "Custom Maze", "Other"
        ])
        form_layout.addRow("Type:", self.generation_combo)
        
        self.description_edit = QTextEdit()
        self.description_edit.setPlaceholderText("Optional description or notes about this maze")
        self.description_edit.setMaximumHeight(80)
        form_layout.addRow("Description:", self.description_edit)
        
        # Maze info (read-only)
        info_layout = QHBoxLayout()
        self.size_label = QLabel()
        self.walls_label = QLabel()
        info_layout.addWidget(self.size_label)
        info_layout.addWidget(self.walls_label)
        info_layout.addStretch()
        form_layout.addRow("Info:", info_layout)
        
        layout.addLayout(form_layout)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def _populate_fields(self):
        """Populate form fields with maze data."""
        # Set default name based on maze properties
        if not self.maze_data.name:
            self.name_edit.setText(f"Maze {self.maze_data.width}x{self.maze_data.height}")
        else:
            self.name_edit.setText(self.maze_data.name)
        
        # Set generation method if available
        if self.maze_data.generation_method:
            index = self.generation_combo.findText(self.maze_data.generation_method)
            if index >= 0:
                self.generation_combo.setCurrentIndex(index)
        
        # Set description
        self.description_edit.setText(self.maze_data.description)
        
        # Display maze info
        self.size_label.setText(f"Size: {self.maze_data.width}×{self.maze_data.height}")
        self.walls_label.setText(f"Walls: {len(self.maze_data.walls)}")
    
    def get_maze_data(self) -> MazeData:
        """Get the maze data with updated metadata."""
        self.maze_data.name = self.name_edit.text().strip()
        self.maze_data.generation_method = self.generation_combo.currentText()
        self.maze_data.description = self.description_edit.toPlainText().strip()
        return self.maze_data


class MazeManagerDialog(QDialog):
    """Dialog for managing saved mazes - loading, deleting, etc."""
    
    maze_selected = Signal(MazeData)  # Emitted when a maze is selected for loading
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Maze Manager")
        self.setModal(True)
        self.resize(800, 600)
        
        self.selected_maze_data = None
        self.selected_filepath = None
        
        self._setup_ui()
        self._refresh_maze_list()
    
    def _setup_ui(self):
        """Setup the UI elements."""
        layout = QHBoxLayout(self)
        
        # Create splitter for resizable panes
        splitter = QSplitter(Qt.Horizontal)
        
        # Left pane - maze list
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        left_layout.addWidget(QLabel("Saved Mazes:"))
        
        self.maze_list = QListWidget()
        self.maze_list.currentItemChanged.connect(self._on_maze_selected)
        left_layout.addWidget(self.maze_list)
        
        # Buttons for maze list
        list_buttons_layout = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_maze_list)
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self._delete_selected_maze)
        self.delete_btn.setEnabled(False)
        
        list_buttons_layout.addWidget(self.refresh_btn)
        list_buttons_layout.addWidget(self.delete_btn)
        list_buttons_layout.addStretch()
        left_layout.addLayout(list_buttons_layout)
        
        splitter.addWidget(left_widget)
        
        # Right pane - maze details
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Maze details group
        details_group = QGroupBox("Maze Details")
        details_layout = QFormLayout(details_group)
        
        self.name_label = QLabel("No maze selected")
        self.size_label = QLabel("-")
        self.walls_label = QLabel("-")
        self.type_label = QLabel("-")
        self.created_label = QLabel("-")
        self.description_label = QTextEdit()
        self.description_label.setReadOnly(True)
        self.description_label.setMaximumHeight(80)
        
        details_layout.addRow("Name:", self.name_label)
        details_layout.addRow("Size:", self.size_label)
        details_layout.addRow("Walls:", self.walls_label)
        details_layout.addRow("Type:", self.type_label)
        details_layout.addRow("Created:", self.created_label)
        details_layout.addRow("Description:", self.description_label)
        
        right_layout.addWidget(details_group)
        right_layout.addStretch()
        
        splitter.addWidget(right_widget)
        
        # Set splitter proportions
        splitter.setSizes([400, 400])
        layout.addWidget(splitter)
        
        # Bottom buttons
        button_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("Load Maze")
        self.load_btn.clicked.connect(self._load_selected_maze)
        self.load_btn.setEnabled(False)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(self.load_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.close_btn)
        
        # Add button layout to main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(splitter)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)
    
    def _refresh_maze_list(self):
        """Refresh the list of saved mazes."""
        self.maze_list.clear()
        
        saved_mazes = list_saved_mazes()
        
        if not saved_mazes:
            item = QListWidgetItem("No saved mazes found")
            item.setData(Qt.UserRole, None)
            item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
            self.maze_list.addItem(item)
            return
        
        for filepath, maze_data in saved_mazes:
            # Create display text
            name = maze_data.name or f"Maze {maze_data.width}x{maze_data.height}"
            size_info = f"{maze_data.width}×{maze_data.height}"
            walls_info = f"{len(maze_data.walls)} walls"
            
            display_text = f"{name} ({size_info}, {walls_info})"
            
            item = QListWidgetItem(display_text)
            item.setData(Qt.UserRole, (filepath, maze_data))
            self.maze_list.addItem(item)
    
    def _on_maze_selected(self, current, previous):
        """Handle maze selection."""
        if current is None:
            self._clear_maze_details()
            self.load_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)
            return
        
        data = current.data(Qt.UserRole)
        if data is None:
            self._clear_maze_details()
            self.load_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)
            return
        
        filepath, maze_data = data
        self.selected_filepath = filepath
        self.selected_maze_data = maze_data
        
        self._update_maze_details(maze_data)
        self.load_btn.setEnabled(True)
        self.delete_btn.setEnabled(True)
    
    def _clear_maze_details(self):
        """Clear the maze details display."""
        self.name_label.setText("No maze selected")
        self.size_label.setText("-")
        self.walls_label.setText("-")
        self.type_label.setText("-")
        self.created_label.setText("-")
        self.description_label.setText("")
    
    def _update_maze_details(self, maze_data: MazeData):
        """Update the maze details display."""
        from datetime import datetime
        
        self.name_label.setText(maze_data.name or "Unnamed Maze")
        self.size_label.setText(f"{maze_data.width}×{maze_data.height}")
        self.walls_label.setText(str(len(maze_data.walls)))
        self.type_label.setText(maze_data.generation_method or "Unknown")
        
        # Format creation date
        try:
            created_dt = datetime.fromisoformat(maze_data.created_at)
            self.created_label.setText(created_dt.strftime("%Y-%m-%d %H:%M:%S"))
        except:
            self.created_label.setText("Unknown")
        
        self.description_label.setText(maze_data.description or "No description")
    
    def _load_selected_maze(self):
        """Load the selected maze."""
        if self.selected_maze_data is None:
            return
        
        self.maze_selected.emit(self.selected_maze_data)
        self.accept()
    
    def _delete_selected_maze(self):
        """Delete the selected maze after confirmation."""
        if not self.selected_filepath or not self.selected_maze_data:
            return
        
        maze_name = self.selected_maze_data.name or "this maze"
        
        reply = QMessageBox.question(
            self, "Confirm Deletion",
            f"Are you sure you want to delete '{maze_name}'?\n\nThis action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            if delete_maze(self.selected_filepath):
                self._refresh_maze_list()
                self._clear_maze_details()
                QMessageBox.information(self, "Success", "Maze deleted successfully!")
            else:
                QMessageBox.warning(self, "Error", "Failed to delete maze file.")


def show_save_maze_dialog(grid, start_coord=None, target_coord=None, 
                         generation_method="", parent=None) -> bool:
    """Show save maze dialog and save the maze if confirmed."""
    # Extract maze data from grid
    maze_data = extract_maze_from_grid(grid, start_coord, target_coord)
    maze_data.generation_method = generation_method
    
    # Show save dialog
    dialog = SaveMazeDialog(maze_data, parent)
    if dialog.exec() == QDialog.Accepted:
        # Get updated maze data with user input
        updated_maze_data = dialog.get_maze_data()
        
        # Generate filename and save
        filepath = generate_maze_filename(updated_maze_data)
        if save_maze(updated_maze_data, filepath):
            QMessageBox.information(
                parent, "Success", 
                f"Maze '{updated_maze_data.name}' saved successfully!"
            )
            return True
        else:
            QMessageBox.warning(
                parent, "Error", 
                "Failed to save maze. Please check file permissions."
            )
    
    return False


def show_maze_manager_dialog(parent=None) -> MazeData:
    """Show maze manager dialog and return selected maze data."""
    dialog = MazeManagerDialog(parent)
    if dialog.exec() == QDialog.Accepted:
        return dialog.selected_maze_data
    return None