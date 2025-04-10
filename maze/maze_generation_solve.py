import heapq  # For priority queue used in A* algorithm
import tkinter as tk  # For drawing the maze GUI
import tkinter.messagebox as msgbox  # For showing final stats popup
import random  # For generating random maze
import time  # For timing how long solving the maze takes

# Heuristic function: Manhattan distance between two point
def solve(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# A* algorithm to find shortest path in the maze
def astar(maze, start, end):
    rows, cols = len(maze), len(maze[0])
    open_set = [(0, start)]  # Priority queue with (f_score, node)
    g = {start: 0}  # Cost from start to each node
    f = {start: solve(start, end)}  # Total estimated cost (g + heuristic)
    init = {}  # To reconstruct the path
    visited_order = []  # Order of visited nodes for animation

    while open_set:
        _, curr = heapq.heappop(open_set)  # Get node with lowest f

        if curr == end:
            # Reconstruct path if goal is reached
            path = []
            
            while curr in init:
                path.append(curr)
                curr = init[curr]
            
            return path[::-1], visited_order  # Reverse to get correct order

        if curr in visited_order:
            continue  # Skip already visited nodes

        visited_order.append(curr)
        x, y = curr

        # Explore 4 directions: up, down, left, right
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next = (x + dx, y + dy)

            # Check bounds and make sure the next cell is not a wall
            if 0 <= next[0] < rows and 0 <= next[1] < cols and maze[next[0]][next[1]] == 0:
                temp = g[curr] + 1  # Tentative g score

                # If this is a better path to next or first time visiting
                if next not in g or temp < g[next]:
                    g[next] = temp
                    f[next] = temp + solve(next, end)  # f = g + h
                    heapq.heappush(open_set, (f[next], next))  # Add to queue
                    init[next] = curr  # Record the parent

    return [], visited_order  # If no path is found

# Function to draw and animate the maze solving process
def draw_maze(maze, path, size, start, end, visited, solve_time):
    max_width = 1920  # Max canvas width
    rows, cols = len(maze), len(maze[0])

    # Determine cell size based on maze size
    cell_width = max_width // (rows + (size * 0.6))
    cell_height = int(cell_width / 1.6)
    tot_height = cell_height * rows  # Total height of canvas

    root = tk.Tk()
    root.title("MAZE SOLVER")
    
    # Create canvas to draw maze
    canvas = tk.Canvas(root, width=max_width, height=tot_height, bg="white")
    canvas.pack()

    # Draw the maze grid
    for y in range(rows):
        for x in range(cols):
            if maze[y][x] == 1:
                # Draw wall as black rectangle
                canvas.create_rectangle(
                    x * cell_width, y * cell_height, (x + 1) * cell_width, (y + 1) * cell_height,
                    fill="black", outline="black"
                )

    # Draw start (green) and end (red) cells
    sx, sy = start[1], start[0]
    ex, ey = end[1], end[0]
    
    canvas.create_rectangle(sx * cell_width, sy * cell_height, (sx + 1) * cell_width, (sy + 1) * cell_height, fill="green", outline="green")
    canvas.create_rectangle(ex * cell_width, ey * cell_height, (ex + 1) * cell_width, (ey + 1) * cell_height, fill="red", outline="red")

    # Animate visiting nodes one by one
    def animate_visited(i=0):
        if i < len(visited):
            y, x = visited[i]
            
            if (y, x) != start and (y, x) != end:
                # Draw visited node as yellow
                canvas.create_rectangle(
                    x * cell_width, y * cell_height, (x + 1) * cell_width, (y + 1) * cell_height,
                    fill="yellow", outline="yellow"
                )
            
            root.after(5, lambda: animate_visited(i + 1))  # Animate next step
        else:
            animate_path()  # Once done, animate path

    # Animate final path from start to end
    def animate_path(index=0):
        if index < len(path):
            y, x = path[index]
            
            if (y, x) != end:
                # Draw path as blue
                canvas.create_rectangle(
                    x * cell_width, y * cell_height, (x + 1) * cell_width, (y + 1) * cell_height,
                    fill="blue", outline="blue"
                )
            
            root.after(20, lambda: animate_path(index + 1))  # Animate next path step
        else:
            show_stats()  # Show stats when done

    # Show a popup with solving statistics
    def show_stats():
        msgbox.showinfo(
            "Maze Solver Stats",
            f"Visited nodes: {len(visited)}\nPath length: {len(path)}\nSolve time: {solve_time:.3f} seconds"
        )

    animate_visited()  # Start animation
    root.mainloop()  # Run the GUI window

# Generate a random maze using DFS (depth-first search)
def gen_maze(size):
    maze = [[1] * (2 * size + 1) for _ in range(2 * size + 1)]  # Fill maze with walls
    seen = [[False] * size for _ in range(size)]  # Track visited maze cells
    stack = [(0, 0)]  # DFS stack
    seen[0][0] = True

    while stack:
        x, y = stack[-1]
        maze[2 * y + 1][2 * x + 1] = 0  # Carve out path
        dir = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Directions: up, right, down, left
        random.shuffle(dir)  # Randomize direction order
        found = False

        for dx, dy in dir:
            nx, ny = x + dx, y + dy

            if 0 <= nx < size and 0 <= ny < size and not seen[ny][nx]:
                # Carve wall between current and next cell
                maze[2 * y + 1 + dy][2 * x + 1 + dx] = 0
                seen[ny][nx] = True
                stack.append((nx, ny))  # Visit next cell
                found = True
                break

        if not found:
            stack.pop()  # Backtrack if no unvisited neighbors

    return maze  # Return generated maze

# MAIN PROGRAM
size = int(input("Enter maze size: "))  # User input
maze = gen_maze(size)  # Generate maze
start = (1, 1)  # Start position
end = (len(maze) - 2, len(maze[0]) - 2)  # End position (bottom right)

start_time = time.time()
path, visited = astar(maze, start, end)  # Solve maze
solve_time = time.time() - start_time  # Time taken to solve

draw_maze(maze, path, size, start, end, visited, solve_time)  # Animate result
