#!/usr/bin/env python3
"""
Analyze a maze to determine if there's a valid path and identify potential issues.
"""
import json
import sys
from collections import deque
from pathlib import Path

def load_maze(filepath):
    """Load maze data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def visualize_maze(width, height, walls, start, target):
    """Create a visual representation of the maze."""
    # Create grid
    grid = [['.' for _ in range(width)] for _ in range(height)]
    
    # Add walls
    wall_set = set(tuple(wall) for wall in walls)
    for y in range(height):
        for x in range(width):
            if (x, y) in wall_set:
                grid[y][x] = '█'
    
    # Add start and target
    grid[start[1]][start[0]] = 'S'
    grid[target[1]][target[0]] = 'T'
    
    # Print maze
    print(f"Maze {width}x{height}:")
    print("Start (S) at", start, "Target (T) at", target)
    print()
    for row in grid:
        print(''.join(row))
    print()

def find_path_bfs(width, height, walls, start, target):
    """Use BFS to find if a path exists from start to target."""
    wall_set = set(tuple(wall) for wall in walls)
    
    # BFS setup
    queue = deque([tuple(start)])
    visited = {tuple(start)}
    parent = {}
    
    # Directions: up, down, left, right
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    
    while queue:
        current = queue.popleft()
        
        # Check if we reached target
        if current == tuple(target):
            # Reconstruct path
            path = []
            node = current
            while node in parent:
                path.append(node)
                node = parent[node]
            path.append(tuple(start))
            path.reverse()
            return True, path
        
        # Explore neighbors
        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            
            # Check bounds
            if 0 <= nx < width and 0 <= ny < height:
                neighbor = (nx, ny)
                
                # Check if not a wall and not visited
                if neighbor not in wall_set and neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)
    
    return False, []

def analyze_connectivity(width, height, walls, start):
    """Analyze which areas of the maze are reachable from start."""
    wall_set = set(tuple(wall) for wall in walls)
    
    # BFS to find all reachable positions
    queue = deque([tuple(start)])
    reachable = {tuple(start)}
    
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    
    while queue:
        current = queue.popleft()
        
        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            
            if 0 <= nx < width and 0 <= ny < height:
                neighbor = (nx, ny)
                
                if neighbor not in wall_set and neighbor not in reachable:
                    reachable.add(neighbor)
                    queue.append(neighbor)
    
    return reachable

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_maze.py <maze_file.json>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    try:
        maze_data = load_maze(filepath)
    except Exception as e:
        print(f"Error loading maze: {e}")
        sys.exit(1)
    
    width = maze_data['width']
    height = maze_data['height']
    walls = maze_data['walls']
    start = maze_data['start']
    target = maze_data['target']
    
    print(f"Analyzing maze: {maze_data['name']}")
    print(f"Dimensions: {width}x{height}")
    print(f"Wall count: {len(walls)}")
    print(f"Start: {start}")
    print(f"Target: {target}")
    print()
    
    # Visualize maze
    visualize_maze(width, height, walls, start, target)
    
    # Check path existence
    has_path, path = find_path_bfs(width, height, walls, start, target)
    
    if has_path:
        print(f"✅ Path exists! Length: {len(path)} steps")
        print("Path:", " -> ".join(f"({x},{y})" for x, y in path[:10]) + ("..." if len(path) > 10 else ""))
    else:
        print("❌ No path exists from start to target!")
        
        # Analyze connectivity
        reachable = analyze_connectivity(width, height, walls, start)
        total_free_spaces = width * height - len(walls)
        
        print(f"Reachable spaces from start: {len(reachable)}")
        print(f"Total free spaces: {total_free_spaces}")
        print(f"Disconnected spaces: {total_free_spaces - len(reachable)}")
        
        # Check if target is reachable
        if tuple(target) not in reachable:
            print(f"🔍 Target {target} is in a disconnected area!")
        
        # Find which connected component the target is in
        target_reachable = analyze_connectivity(width, height, walls, target)
        print(f"Spaces reachable from target: {len(target_reachable)}")
        
        # Wall density analysis
        wall_density = len(walls) / (width * height) * 100
        print(f"Wall density: {wall_density:.1f}%")
        
        if wall_density > 60:
            print("⚠️  High wall density might make the maze very difficult")

if __name__ == "__main__":
    main()