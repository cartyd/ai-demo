"""SVG cheese widget for the A* visualizer."""

import os
from PySide6.QtWidgets import QGraphicsItem
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtGui import QPainter
from PySide6.QtCore import QRectF, Qt


class SvgCheese(QGraphicsItem):
    """
    SVG cheese graphics item for target visualization.
    """
    
    def __init__(self, size: int = 32):
        super().__init__()
        self._size = size
        self._svg_renderer = None
        self._load_svg()
    
    def _load_svg(self):
        """Load the cheese SVG file."""
        # Get path to cheese.svg
        current_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(os.path.dirname(current_dir), "assets")
        svg_path = os.path.join(assets_dir, "cheese.svg")
        
        if os.path.exists(svg_path):
            self._svg_renderer = QSvgRenderer(svg_path)
        else:
            # Fallback: create a simple renderer with inline SVG
            svg_content = """
            <svg width="32" height="32" viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg">
              <path d="M 4 28 L 28 28 L 28 12 Q 26 8 22 6 Q 18 4 14 6 Q 10 8 8 12 L 4 28 Z" 
                    fill="#FFD700" stroke="#DAA520" stroke-width="1.5"/>
              <ellipse cx="12" cy="20" rx="2" ry="1.5" fill="#FFA500"/>
              <ellipse cx="20" cy="16" rx="1.5" ry="1.2" fill="#FFA500"/>
              <ellipse cx="16" cy="23" rx="1.8" ry="1.3" fill="#FFA500"/>
              <circle cx="18" cy="18" r="0.8" fill="#FFA500"/>
            </svg>
            """
            self._svg_renderer = QSvgRenderer(svg_content.encode())
    
    def boundingRect(self) -> QRectF:
        """Return the bounding rectangle of the item."""
        return QRectF(0, 0, self._size, self._size)
    
    def paint(self, painter: QPainter, option, widget=None):
        """Paint the cheese SVG."""
        if self._svg_renderer and self._svg_renderer.isValid():
            # Set high quality rendering
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            
            # Render the SVG
            self._svg_renderer.render(painter, self.boundingRect())
        else:
            # Fallback: draw a simple triangle
            painter.setBrush(Qt.yellow)
            painter.setPen(Qt.darkYellow)
            rect = self.boundingRect()
            # Draw a cheese-like triangle
            points = [
                rect.topLeft() + QPointF(rect.width() * 0.2, rect.height() * 0.8),
                rect.topRight() + QPointF(-rect.width() * 0.1, rect.height() * 0.3),
                rect.bottomRight() + QPointF(0, -rect.height() * 0.1),
                rect.bottomLeft()
            ]
            painter.drawPolygon(points)
    
    def set_position(self, x: float, y: float):
        """Set the position of the cheese."""
        self.setPos(x - self._size // 2, y - self._size // 2)
    
    def get_size(self) -> int:
        """Get the size of the cheese sprite."""
        return self._size
    
    def set_size(self, size: int):
        """Set the size of the cheese sprite."""
        self._size = size
        self.prepareGeometryChange()
        self.update()