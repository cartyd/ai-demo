"""Grid view for A* pathfinding visualization."""

from typing import Dict, Optional, Tuple
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene
from PySide6.QtGui import QPainter
from PySide6.QtCore import Qt, Signal

from ..domain.types import Grid, Coord
from ..app.controller import AStarController
from .tiles import GridTile
from .svg_mouse import SvgMouse
from .svg_cheese import SvgCheese


class GridView(QGraphicsView):
    """Graphics view for displaying and interacting with the A* grid."""
    
    # Signals
    tile_clicked = Signal(int, int)  # x, y coordinates
    
    def __init__(self, controller: AStarController):
        super().__init__()
        
        self.controller = controller
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        
        # Visual components
        self.tiles: Dict[Tuple[int, int], GridTile] = {}
        self.mouse_sprite: Optional[SvgMouse] = None
        self.cheese_sprite: Optional[SvgCheese] = None
        
        # Settings
        self.tile_size = 25.0
        self.show_costs = False
        self.edit_mode = "wall"  # "wall", "start", "target"
        
        # Setup view
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        
        # Connect controller signals
        self.controller.grid_updated.connect(self.update_grid)
        
        # Initial grid update
        self.update_grid()
    
    def update_grid(self):
        """Update the visual grid from the controller's grid."""
        grid = self.controller.grid
        if not grid:
            return
        
        # Clear existing items
        self.scene.clear()
        self.tiles.clear()
        self.mouse_sprite = None
        self.cheese_sprite = None
        
        # Calculate scene size
        scene_width = grid.width * self.tile_size
        scene_height = grid.height * self.tile_size
        self.scene.setSceneRect(0, 0, scene_width, scene_height)
        
        # Create tiles
        for y in range(grid.height):
            for x in range(grid.width):
                coord = (x, y)
                node = grid.get_node(coord)
                if node:
                    tile = GridTile(x, y, self.tile_size, node)
                    tile.set_show_costs(self.show_costs)
                    self.scene.addItem(tile)
                    self.tiles[(x, y)] = tile
        
        # Add mouse sprite if start position exists
        start_coord = self.controller.start_coord
        if start_coord:
            self.mouse_sprite = SvgMouse(int(self.tile_size * 0.8))
            self.scene.addItem(self.mouse_sprite)
            self._update_mouse_position()
        
        # Add cheese sprite if target position exists
        target_coord = self.controller.target_coord
        if target_coord:
            self.cheese_sprite = SvgCheese(int(self.tile_size * 0.8))
            self.scene.addItem(self.cheese_sprite)
            self._update_cheese_position()
        
        # Update tile appearances
        self._update_tile_appearances()
    
    def _update_tile_appearances(self):
        """Update all tile visual appearances."""
        for tile in self.tiles.values():
            tile.update_appearance()
    
    def _update_mouse_position(self):
        """Update mouse sprite position."""
        if self.mouse_sprite and self.controller.start_coord:
            x, y = self.controller.start_coord
            center_x = (x + 0.5) * self.tile_size
            center_y = (y + 0.5) * self.tile_size
            self.mouse_sprite.set_position(center_x, center_y)
    
    def _update_cheese_position(self):
        """Update cheese sprite position."""
        if self.cheese_sprite and self.controller.target_coord:
            x, y = self.controller.target_coord
            center_x = (x + 0.5) * self.tile_size
            center_y = (y + 0.5) * self.tile_size
            self.cheese_sprite.set_position(center_x, center_y)
    
    def set_show_costs(self, show: bool):
        """Enable or disable cost display on all tiles."""
        self.show_costs = show
        for tile in self.tiles.values():
            tile.set_show_costs(show)
    
    def set_edit_mode(self, mode: str):
        """Set the current edit mode."""
        self.edit_mode = mode
    
    def mousePressEvent(self, event):
        """Handle mouse press events for tile editing."""
        if event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            x = int(scene_pos.x() // self.tile_size)
            y = int(scene_pos.y() // self.tile_size)
            
            grid = self.controller.grid
            if grid and grid.is_valid_coord((x, y)):
                self._handle_tile_click(x, y)
        
        super().mousePressEvent(event)
    
    def _handle_tile_click(self, x: int, y: int):
        """Handle clicking on a tile."""
        coord = (x, y)
        
        if self.edit_mode == "wall":
            # Toggle wall
            node = self.controller.grid.get_node(coord)
            if node and node.state not in ["start", "target"]:
                new_state = "empty" if node.state == "wall" else "wall"
                self.controller.set_node_state(coord, new_state)
        
        elif self.edit_mode == "start":
            # Set start position
            self.controller.set_node_state(coord, "start")
        
        elif self.edit_mode == "target":
            # Set target position
            self.controller.set_node_state(coord, "target")
    
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming."""
        zoom_factor = 1.15
        if event.angleDelta().y() > 0:
            self.scale(zoom_factor, zoom_factor)
        else:
            self.scale(1 / zoom_factor, 1 / zoom_factor)
    
    def fit_in_view(self):
        """Fit the entire grid in the view."""
        if self.scene.items():
            self.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
    
    def reset_zoom(self):
        """Reset zoom to 1:1."""
        self.resetTransform()