"""Rotatable SVG mouse widget for the A* visualizer."""

import os
from typing import Optional
from PySide6.QtWidgets import QGraphicsItem, QGraphicsPixmapItem
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtGui import QPainter, QPixmap, QTransform
from PySide6.QtCore import QRectF, Qt


class SvgMouse(QGraphicsItem):
    """
    SVG mouse graphics item that can rotate to face movement direction.
    """
    
    def __init__(self, size: int = 32):
        super().__init__()
        self._size = size
        self._rotation = 0.0  # Current rotation in degrees
        self._svg_renderer = None
        self._load_svg()
        
        # Enable transformations
        self.setTransformOriginPoint(size // 2, size // 2)
    
    def _load_svg(self):
        """Load the mouse SVG file."""
        # Get path to mouse.svg
        current_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(os.path.dirname(current_dir), "assets")
        svg_path = os.path.join(assets_dir, "mouse.svg")
        
        if os.path.exists(svg_path):
            self._svg_renderer = QSvgRenderer(svg_path)
        else:
            # Fallback: create a simple renderer with inline SVG
            svg_content = """
            <svg width="32" height="32" viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg">
              <ellipse cx="16" cy="18" rx="10" ry="8" fill="#8B4513" stroke="#654321" stroke-width="1"/>
              <ellipse cx="16" cy="10" rx="6" ry="5" fill="#CD853F" stroke="#8B4513" stroke-width="1"/>
              <ellipse cx="16" cy="6" rx="1.5" ry="1" fill="#FFB6C1" stroke="#8B4513" stroke-width="0.5"/>
              <circle cx="13" cy="9" r="1.5" fill="#000"/>
              <circle cx="19" cy="9" r="1.5" fill="#000"/>
              <circle cx="13.5" cy="8.5" r="0.5" fill="#FFF"/>
              <circle cx="19.5" cy="8.5" r="0.5" fill="#FFF"/>
            </svg>
            """
            self._svg_renderer = QSvgRenderer(svg_content.encode())
    
    def set_rotation(self, degrees: float):
        """Set the rotation angle in degrees."""
        self._rotation = degrees % 360
        self.setRotation(self._rotation)
        self.update()
    
    def get_rotation(self) -> float:
        """Get the current rotation angle in degrees."""
        return self._rotation
    
    def boundingRect(self) -> QRectF:
        """Return the bounding rectangle of the item."""
        return QRectF(0, 0, self._size, self._size)
    
    def paint(self, painter: QPainter, option, widget=None):
        """Paint the mouse SVG."""
        if self._svg_renderer and self._svg_renderer.isValid():
            # Set high quality rendering
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            
            # Render the SVG
            self._svg_renderer.render(painter, self.boundingRect())
        else:
            # Fallback: draw a simple circle
            painter.setBrush(Qt.brown)
            painter.setPen(Qt.darkGray)
            rect = self.boundingRect()
            painter.drawEllipse(rect.adjusted(2, 2, -2, -2))
    
    def set_position(self, x: float, y: float):
        """Set the position of the mouse."""
        self.setPos(x - self._size // 2, y - self._size // 2)
    
    def get_size(self) -> int:
        """Get the size of the mouse sprite."""
        return self._size
    
    def set_size(self, size: int):
        """Set the size of the mouse sprite."""
        self._size = size
        self.setTransformOriginPoint(size // 2, size // 2)
        self.prepareGeometryChange()
        self.update()
    
    def animate_to_direction(self, direction: tuple[int, int]):
        """Animate rotation to face the given direction vector."""
        from ..domain.neighbors import get_rotation_angle
        target_angle = get_rotation_angle(direction)
        self.set_rotation(target_angle)
