"""Grid tiles for RL pathfinding visualization."""

import numpy as np
from typing import Optional, Dict
from PySide6.QtWidgets import QGraphicsItem, QGraphicsRectItem, QGraphicsTextItem
from PySide6.QtGui import QBrush, QPen, QColor, QFont
from PySide6.QtCore import QRectF, Qt

from ..domain.types import GridNode, NodeState, ACTION_TO_INT


class GridTile(QGraphicsRectItem):
    """Graphics item representing a single grid tile with RL visualization."""
    
    def __init__(self, x: int, y: int, size: float, node: GridNode):
        super().__init__(0, 0, size, size)
        self.grid_x = x
        self.grid_y = y
        self.size = size
        self.node = node
        self.show_costs = False
        self.show_q_values = False
        
        # Position the tile
        self.setPos(x * size, y * size)
        
        # Create text items for displaying values
        self._cost_text = QGraphicsTextItem(parent=self)
        self._q_value_texts: Dict[str, QGraphicsTextItem] = {}
        
        # Setup text items
        self._setup_text_items()
        
        # Initial appearance update
        self.update_appearance()
    
    def _setup_text_items(self):
        """Setup text items for displaying costs and Q-values."""
        font = QFont("Arial", int(self.size * 0.15))
        
        # Cost text (center)
        self._cost_text.setFont(font)
        cost_rect = self._cost_text.boundingRect()
        self._cost_text.setPos(
            (self.size - cost_rect.width()) / 2,
            (self.size - cost_rect.height()) / 2
        )
        
        # Q-value texts (in corners)
        small_font = QFont("Arial", int(self.size * 0.12))
        positions = {
            "up": (self.size * 0.5, self.size * 0.1),      # Top center
            "left": (self.size * 0.1, self.size * 0.5),    # Left center
            "right": (self.size * 0.85, self.size * 0.5),  # Right center
            "down": (self.size * 0.5, self.size * 0.85)    # Bottom center
        }
        
        for action, pos in positions.items():
            text_item = QGraphicsTextItem(parent=self)
            text_item.setFont(small_font)
            text_item.setPos(pos[0], pos[1])
            self._q_value_texts[action] = text_item
    
    def update_appearance(self):
        """Update tile appearance based on node state and values."""
        colors = self._get_state_colors()
        brush_color, pen_color = colors
        
        # Set brush and pen
        self.setBrush(QBrush(brush_color))
        self.setPen(QPen(pen_color, 1))
        
        # Update text displays
        self._update_cost_display()
        self._update_q_value_display()
    
    def _get_state_colors(self) -> tuple[QColor, QColor]:
        """Get colors for current node state."""
        state = self.node.state
        
        # Base colors for different states
        color_map = {
            "empty": (QColor(240, 240, 240), QColor(180, 180, 180)),
            "wall": (QColor(60, 60, 60), QColor(40, 40, 40)),
            "start": (QColor(100, 255, 100), QColor(50, 200, 50)),
            "target": (QColor(255, 100, 100), QColor(200, 50, 50)),
            "visited": (QColor(200, 200, 255), QColor(150, 150, 200)),
            "current": (QColor(255, 255, 100), QColor(200, 200, 50)),
            "path": (QColor(255, 200, 100), QColor(200, 150, 50)),
            "optimal_path": (QColor(255, 255, 100), QColor(200, 200, 50)),  # Yellow for validated optimal path
            # Training visualization states
            "training_current": (QColor(255, 150, 255), QColor(200, 100, 200)),  # Magenta for current training position
            "training_visited": (QColor(220, 180, 255), QColor(170, 130, 200)),  # Light purple for visited during training
            "training_considering": (QColor(255, 220, 150), QColor(200, 170, 100)),  # Light orange for considering moves
        }
        
        # Q-value visualization colors (for empty tiles)
        if state == "empty" and self.show_q_values:
            max_q = self.node.q_values.max_value()
            if max_q > 0:
                # Scale color based on Q-value magnitude
                intensity = min(abs(max_q) / 10.0, 1.0)  # Normalize to [0, 1]
                if max_q > 0:
                    # Green for positive Q-values
                    green = int(255 * intensity)
                    return (QColor(200, 255, 200, green), QColor(100, 200, 100))
                else:
                    # Red for negative Q-values  
                    red = int(255 * intensity)
                    return (QColor(255, 200, 200, red), QColor(200, 100, 100))
        
        return color_map.get(state, (QColor(240, 240, 240), QColor(180, 180, 180)))
    
    def _update_cost_display(self):
        """Update cost text display."""
        if self.show_costs and self.node.visit_count > 0:
            cost_text = f"V:{self.node.visit_count}\nR:{self.node.last_reward:.1f}"
            self._cost_text.setPlainText(cost_text)
            self._cost_text.setVisible(True)
            
            # Center the text
            rect = self._cost_text.boundingRect()
            self._cost_text.setPos(
                (self.size - rect.width()) / 2,
                (self.size - rect.height()) / 2
            )
        else:
            self._cost_text.setVisible(False)
    
    def _update_q_value_display(self):
        """Update Q-value text display."""
        if self.show_q_values and self.node.state in ["empty", "start", "target", "path", "optimal_path"]:
            q_values = self.node.q_values
            
            # Display Q-values for each action
            q_dict = {
                "up": q_values.up,
                "down": q_values.down,
                "left": q_values.left,
                "right": q_values.right
            }
            
            for action, value in q_dict.items():
                text_item = self._q_value_texts[action]
                if abs(value) > 0.01:  # Only show significant values
                    text_item.setPlainText(f"{value:.1f}")
                    text_item.setVisible(True)
                    
                    # Color based on value
                    if value > 0:
                        text_item.setDefaultTextColor(QColor(0, 150, 0))  # Green for positive
                    else:
                        text_item.setDefaultTextColor(QColor(150, 0, 0))  # Red for negative
                else:
                    text_item.setVisible(False)
        else:
            # Hide all Q-value texts
            for text_item in self._q_value_texts.values():
                text_item.setVisible(False)
    
    def set_show_costs(self, show: bool):
        """Enable or disable cost display."""
        self.show_costs = show
        self._update_cost_display()
    
    def set_show_q_values(self, show: bool):
        """Enable or disable Q-value display."""
        self.show_q_values = show
        self._update_q_value_display()
        # Also update tile colors since they depend on Q-values
        self.update_appearance()
    
    def mousePressEvent(self, event):
        """Handle mouse press events."""
        # Let the parent view handle the click
        super().mousePressEvent(event)