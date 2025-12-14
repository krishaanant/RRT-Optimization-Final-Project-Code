import numpy as np 
import random
from rtree import index
import uuid





def obstacle_generator(obstacles):
    """
    Add obstacles to r-tree
    :param obstacles: list of obstacles
    """
    for obstacle in obstacles:
        yield (uuid.uuid4().int, obstacle, obstacle)
        
        
def generate_random_obstacles_pct(dimension_lengths, start, end, coverage_percentage, tolerance=0.02):
    """
    Generates n random obstacles such that they cover a given percentage of the workspace within a tolerance.
    
    Args:
        dimension_lengths (list of tuples): [(min_x, max_x), (min_y, max_y), ...] defining workspace size
        start (list): Coordinates of the start position
        end (list): Coordinates of the end position
        n (int): Number of obstacles (maximum), we can have fewer if the covverage percentage is reached
        coverage_percentage (float): Percentage of the workspace to be covered by obstacles (0 to 100)
        tolerance (float): Allowed deviation in coverage percentage (default 0.02 for ±2%)

    Returns:
        List of obstacles, where each obstacle is defined as [min_x, min_y, ..., max_x, max_y, ...].
    """

    p = index.Property()
    dimensions = len(dimension_lengths)
    p.dimension = dimensions
    obs = index.Index(interleaved=True, properties=p)

    # Calculate total workspace volume (area for 2D, volume for 3D)
    workspace_volume = np.prod([dim[1] - dim[0] for dim in dimension_lengths])
    
    # Calculate total target obstacle volume with tolerance
    max_obstacle_volume = ((coverage_percentage + tolerance * 100) / 100.0) * workspace_volume
    target_volume = (coverage_percentage / 100.0) * workspace_volume
    
    obstacles = []
    i = 0
    current_obstacle_volume = 0

    while True:
        center = np.empty(dimensions, float)
        edge_lengths = []
        scollision = True
        fcollision = True
        
        obstacle_volume = 0  # Will be updated

        while True:
            edge_lengths = []
            for j in range(dimensions):
                # max_edge_length = 6  # Adjust max size
                max_edge_length = 12  # Adjust max size
                min_edge_length = 1  # Adjust min size
                
                # max_edge_length = (dimension_lengths[j][1] - dimension_lengths[j][0]) / 5.0  # Adjust max size
                # min_edge_length = (dimension_lengths[j][1] - dimension_lengths[j][0]) / 50.0  # Adjust min size
                
                edge_length = random.uniform(min_edge_length, max_edge_length)
                center[j] = random.uniform(dimension_lengths[j][0] + edge_length,
                                           dimension_lengths[j][1] - edge_length)
                edge_lengths.append(edge_length)

                if abs(start[j] - center[j]) > edge_length:
                    scollision = False
                if abs(end[j] - center[j]) > edge_length:
                    fcollision = False

            obstacle_volume = np.prod(edge_lengths) * 4

            # Ensure that we do not exceed the target coverage
            if current_obstacle_volume + obstacle_volume <= max_obstacle_volume:
                break  # Accept the generated obstacle

        # Define obstacle bounds
        min_corner = np.array([center[j] - edge_lengths[j] for j in range(dimensions)])
        max_corner = np.array([center[j] + edge_lengths[j] for j in range(dimensions)])
        obstacle = np.append(min_corner, max_corner)

        # Check for collisions with existing obstacles
        if len(list(obs.intersection(obstacle))) > 0 or scollision or fcollision:
            continue  # Regenerate obstacle

        # Add obstacle
        i += 1
        current_obstacle_volume += obstacle_volume
        obstacles.append(obstacle)
        obs.insert(i, tuple(obstacle), tuple(obstacle))

        # Stop if we reach the minimum required volume
        if current_obstacle_volume >= target_volume:
            break
        
    # print (f"Current volume: {current_obstacle_volume}, workspace volume: {workspace_volume}")
    return obstacles


class _Cell:
    """A helper class to represent a cell in the maze grid."""
    def __init__(self, r, c):
        self.r = r
        self.c = c
        # Walls are ordered: [Top, Right, Bottom, Left]
        self.walls = [True, True, True, True]
        self.visited = False

    def remove_wall(self, neighbor):
        """Removes the wall between this cell and a neighbor cell."""
        dr = self.r - neighbor.r
        dc = self.c - neighbor.c

        if dr == 1:  # Neighbor is above
            self.walls[0] = False
            neighbor.walls[2] = False
        elif dr == -1:  # Neighbor is below
            self.walls[2] = False
            neighbor.walls[0] = False
        
        if dc == 1:  # Neighbor is to the left
            self.walls[3] = False
            neighbor.walls[1] = False
        elif dc == -1:  # Neighbor is to the right
            self.walls[1] = False
            neighbor.walls[3] = False

def generate_random_maze(width=100, height=100, num_cells_x=10, num_cells_y=10, 
                  wall_thickness=1, min_turning_radius=None, safety_factor=3.0):
    """
    Generates a maze with a guaranteed solution and returns it as a list of obstacles.

    The maze is generated using a recursive backtracking algorithm. Openings are
    created near (0,0) and (width, height) to allow a path from start to end.

    Args:
        width (int): The total width of the environment.
        height (int): The total height of the environment.
        num_cells_x (int): The number of cells in the x-direction (maze complexity).
        num_cells_y (int): The number of cells in the y-direction (maze complexity).
        wall_thickness (float): The thickness of the wall obstacles.
        min_turning_radius (float, optional): If provided, the grid size will be
            calculated to ensure corridors are wide enough for a Dubins vehicle.
            This overrides num_cells_x and num_cells_y.
        safety_factor (float): Multiplier for the turning radius to determine
            corridor width (e.g., 3.0 * radius).

    Returns:
        list: A list of obstacles, where each obstacle is a NumPy array in the
              format [min_x, min_y, max_x, max_y].
    """
    
    # If a turning radius is given, calculate a suitable grid size
    if min_turning_radius is not None:
        corridor_width = safety_factor * min_turning_radius
        if corridor_width >= min(width, height):
            raise ValueError(
                f"Turning radius ({min_turning_radius}) is too large for the world dimensions."
            )
        num_cells_x = int(width / corridor_width)
        num_cells_y = int(height / corridor_width)
        print(f"Dynamically feasible maze: grid size set to {num_cells_x}x{num_cells_y} "
              f"for a turning radius of {min_turning_radius}.")

    # 1. Create a grid of cells using the helper class
    grid = [[_Cell(r, c) for c in range(num_cells_x)] for r in range(num_cells_y)]
    stack = []
    current_cell = grid[0][0]
    current_cell.visited = True
    
    # 2. Use iterative DFS to carve passages
    while True:
        r, c = current_cell.r, current_cell.c
        neighbors = []
        # Find unvisited neighbors
        if r > 0 and not grid[r-1][c].visited: neighbors.append(grid[r-1][c])
        if c < num_cells_x - 1 and not grid[r][c+1].visited: neighbors.append(grid[r][c+1])
        if r < num_cells_y - 1 and not grid[r+1][c].visited: neighbors.append(grid[r+1][c])
        if c > 0 and not grid[r][c-1].visited: neighbors.append(grid[r][c-1])

        if neighbors:
            next_cell = random.choice(neighbors)
            stack.append(current_cell)
            current_cell.remove_wall(next_cell)
            current_cell = next_cell
            current_cell.visited = True
        elif stack:
            current_cell = stack.pop()
        else:
            break # Maze generation complete

    # 3. Convert the maze grid into a list of rectangular obstacles
    obstacles = []
    cell_w = width / num_cells_x
    cell_h = height / num_cells_y
    
    # Add internal walls based on the grid structure
    for r in range(num_cells_y):
        for c in range(num_cells_x):
            # Add right wall if it exists
            if grid[r][c].walls[1] and c < num_cells_x - 1:
                obstacles.append(np.array([
                    (c + 1) * cell_w - wall_thickness / 2, r * cell_h,
                    (c + 1) * cell_w + wall_thickness / 2, (r + 1) * cell_h
                ]))
            # Add bottom wall if it exists
            if grid[r][c].walls[2] and r < num_cells_y - 1:
                obstacles.append(np.array([
                    c * cell_w, (r + 1) * cell_h - wall_thickness / 2,
                    (c + 1) * cell_w, (r + 1) * cell_h + wall_thickness / 2
                ]))

    # Add boundary walls with openings near start (0,0) and end (100,100)
    obstacles.append(np.array([0, height - wall_thickness, width - cell_w, height])) # Top
    obstacles.append(np.array([width - wall_thickness, 0, width, height - cell_h])) # Right
    obstacles.append(np.array([cell_w, 0, width, wall_thickness]))                   # Bottom
    obstacles.append(np.array([0, cell_h, wall_thickness, height]))                   # Left
    
    return obstacles