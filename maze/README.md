
# Maze Solver

## How to Use
Simply run the code and enter the size of the maze that you want to generate.

## Symbols
⬜ represents valid pathway. 

⬛ represents wall.

🟩 represents starting point.

🟥 represents ending point.

🟨 represents explored paths.

🟦 represents the correct path.

## Working
A maze is generated with only one valid answer to it using **Depth-First Search (DFS)** by taking the size of the maze as an input from the user.

The maze is solved using **A\* Algorithm** and then shown and animated using tkinter.

## Concepts Used
### Depth-First Search:

It goes to the deepest point along one path before backtracking and trying another one.

### Manhattan Distance: 
Distance between two points if diagonal movement is disallowed.

### A* Algorithm: 
Pathfinding algorithm to find shortest path from a start point to an end point.

It uses two things: 

*Actual Cost* -> g(n)

*Estimated Cost* -> h(n)

f(n) = g(n) + h(n)

It finds the shortest f(n) and follows that path till you reach the end point.