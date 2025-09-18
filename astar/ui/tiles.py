"""Grid tile graphics items for A* visualization."""

from typing import Optional
from PySide6.QtWidgets import QGraphicsItem, QGraphicsRectItem
from PySide6.QtGui import QPainter, QBrush, QPen, QFont, QColor
from PySide6.QtCore import QRectF, Qt

from ..domain.types import GridNode, NodeState


class GridTile(QGraphicsRectItem):
    """Graphics item representing a single grid tile."""
    
    # Color scheme for different node states
    COLORS = {
        "empty": QColor(240, 240, 240),      # Light gray
        "wall": QColor(64, 64, 64),          # Dark gray  
        "start": QColor(0, 255, 0),          # Green
        "target": QColor(255, 215, 0),       # Gold
        "open": QColor(173, 216, 230),       # Light blue
        "closed": QColor(255, 182, 193),     # Light pink
        "current": QColor(255, 0, 0),        # Red
        "path": QColor(255, 255, 0),         # Yellow
    }
    
    def __init__(self, x: int, y: int, size: float, node: GridNode):
        super().__init__(0, 0, size, size)
        self.grid_x = x
        self.grid_y = y
        self.size = size
        self.node = node
        self.show_costs = False
        
        # Set position
        self.setPos(x * size, y * size)
        
        # Enable hover events
        self.setAcceptHoverEvents(True)
        
        # Initial appearance
        self.update_appearance()
    
    def update_appearance(self):
        """Update the tile appearance based on node state."""
        color = self.COLORS.get(self.node.state, self.COLORS["empty"])
        self.setBrush(QBrush(color))
        
        # Border color
        if self.node.state == "wall":
            self.setPen(QPen(Qt.black, 1))
        else:
            self.setPen(QPen(Qt.gray, 0.5))
    
    def paint(self, painter: QPainter, option, widget=None):
        """Paint the tile with costs if enabled."""
        # Paint the base tile
        super().paint(painter, option, widget)
        
        # Paint costs if enabled and tile is large enough
        if self.show_costs and self.size > 30 and self.node.state not in ["wall", "start", "target"]:
            self._paint_costs(painter)
    
    def _paint_costs(self, painter: QPainter):
        """Paint the G, H, F costs on the tile."""
        rect = self.rect()
        
        # Setup font
        font = QFont("Arial", max(8, int(self.size / 6)))
        painter.setFont(font)
        painter.setPen(Qt.black)
        
        # G cost (top-left)
        if self.node.cost.g > 0:
            g_rect = QRectF(rect.left() + 2, rect.top() + 2, rect.width()/3, rect.height()/3)
            painter.drawText(g_rect, Qt.AlignCenter, f"{self.node.cost.g:.1f}")
        
        # H cost (top-right)  
        if self.node.cost.h > 0:
            h_rect = QRectF(rect.right() - rect.width()/3, rect.top() + 2, rect.width()/3, rect.height()/3)
            painter.drawText(h_rect, Qt.AlignCenter, f"{self.node.cost.h:.1f}")
        
        # F cost (bottom-center)
        if self.node.cost.f > 0:
            f_rect = QRectF(rect.left() + rect.width()/4, rect.bottom() - rect.height()/3, 
                           rect.width()/2, rect.height()/3)
            painter.drawText(f_rect, Qt.AlignCenter, f"{self.node.cost.f:.1f}")
    
    def set_show_costs(self, show: bool):
        """Enable or disable cost display."""
        self.show_costs = show
        self.update()
    
    def hoverEnterEvent(self, event):
        """Handle mouse hover enter."""
        if self.node.state not in ["wall"]:
            # Highlight on hover
            current_brush = self.brush()
            highlight_color = current_brush.color().lighter(120)
            self.setBrush(QBrush(highlight_color))
        super().hoverEnterEvent(event)
    
    def hoverLeaveEvent(self, event):
        """Handle mouse hover leave."""
        self.update_appearance()
        super().hoverLeaveEvent(event)