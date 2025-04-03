# Dynamic Programming: Minimum Path Sum Visualization

📊 A step-by-step animated visualization demonstrating the Dynamic Programming algorithm solving the Minimum Path Sum problem on a 4x4 grid. See the DP table constructed cell-by-cell, with color encoding the minimum path cost.

## 🎥 Visualization Preview

[![Minimum Path Sum DP Visualization](https://img.youtube.com/vi/V8uM64UKcro/0.jpg)](https://youtube.com/shorts/V8uM64UKcro)

**(Click the image above to watch the short video demonstration)**

<https://youtube.com/shorts/V8uM64UKcro>

## 🎯 Project Aim

This project aims to bridge the gap between abstract algorithms and intuitive visual understanding. By visualizing the Dynamic Programming process for the Minimum Path Sum problem, it makes the step-by-step computation tangible and demonstrates how structure emerges from recursive logic using color, order, and motion.

## 🗺️ The Problem: Minimum Path Sum

Given a grid filled with non-negative numbers (costs), find a path from the top-left corner to the bottom-right corner such that the sum of all numbers along the path is minimized. You can only move either **down** or **right** at any point in time.

This visualization uses a 4x4 grid as an example.

## ✨ The Algorithm: Dynamic Programming (DP)

Dynamic Programming solves this problem efficiently by breaking it down into overlapping subproblems. We build a DP table (let's call it `dp_table`) of the same size as the input grid.

-   **`dp_table[i][j]`**: Stores the minimum cost to reach cell `(i, j)` from the starting cell `(0, 0)`.
-   **Base Case**:
    -   `dp_table[0][0]` is the cost of the starting cell itself.
    -   The first row `dp_table[0][j]` accumulates costs from the left.
    -   The first column `dp_table[i][0]` accumulates costs from the top.
-   **Recurrence Relation**: For any other cell `(i, j)`, the minimum cost is calculated as:
    `dp_table[i][j] = cost[i][j] + min(dp_table[i-1][j], dp_table[i][j-1])`
    (The cost of the current cell plus the minimum cost of reaching its top or left neighbor).
-   **Final Answer**: The minimum path sum for the entire grid is found in the bottom-right cell, `dp_table[rows-1][cols-1]`.

## 🎨 The Visualization Explained

The animation demonstrates this process visually:

1.  **Grid Representation**: The 4x4 grid visually represents the `dp_table`.
2.  **Color Intensity**: The color of each cell maps directly to its calculated `dp_table[i][j]` value. Lighter colors represent lower cumulative costs, while darker colors represent higher costs. This provides an intuitive sense of the cost landscape.
3.  **Numerical Value**: The precise `dp_table[i][j]` value is displayed within each cell for accuracy.
4.  **Black Border Highlight**: A thick black border outlines the specific cell whose value is being computed in the current frame/step of the animation, guiding the viewer's focus through the algorithm's progression.
5.  **Step-by-Step Construction**: The animation unfolds sequentially, mirroring the DP algorithm's execution:
    *   Initialization of the start cell.
    *   Filling the first row and column (base cases).
    *   Iteratively calculating the values for the remaining cells based on the recurrence relation.

## 💻 Code

The following Python code using `numpy` and `matplotlib` can be used to calculate the DP table and generate static visualizations similar to the frames in the animation.

```python
import numpy as np
import matplotlib.pyplot as plt

# ---------- 1. Define the grid (Example Cost Grid) ----------
# Note: The exact cost grid used for the video isn't specified,
# this is the one from our previous conversation.
grid = np.array([
    [1, 3, 1, 2],
    [1, 5, 1, 3],
    [4, 2, 1, 1],
    [2, 1, 2, 1]
])

rows, cols = grid.shape
dp = np.zeros_like(grid)
dp[0, 0] = grid[0, 0]

# ---------- 2. Initialize first row and column ----------
# Fill the first column (only reachable by moving down)
for i in range(1, rows):
    dp[i, 0] = dp[i - 1, 0] + grid[i, 0]
# Fill the first row (only reachable by moving right)
for j in range(1, cols):
    dp[0, j] = dp[0, j - 1] + grid[0, j]

# ---------- 3. Fill the DP table ----------
# Calculate minimum cost for each cell based on top or left neighbor
for i in range(1, rows):
    for j in range(1, cols):
        dp[i, j] = min(dp[i - 1, j], dp[i, j - 1]) + grid[i, j]

# ---------- 4. (Optional) Backtrack the minimum path ----------
# This part finds the sequence of cells in the optimal path
path = [(rows - 1, cols - 1)]
i, j = rows - 1, cols - 1
while i > 0 or j > 0:
    if i == 0: # Can only come from left
        j -= 1
    elif j == 0: # Can only come from top
        i -= 1
    # Choose the path from the cell with the smaller cumulative cost
    elif dp[i - 1, j] < dp[i, j - 1]:
        i -= 1 # Came from top
    else:
        j -= 1 # Came from left
    path.append((i, j))
path.reverse() # Path is built backwards, so reverse it

print("DP Table (Minimum Costs):")
print(dp)
print("\nMinimum Path Sum:", dp[rows - 1, cols - 1])
print("\nOptimal Path (Indices):", path)

# ---------- 5. Function to visualize and save a static result ----------
def save_colored_dp_image(dp_table, path, cmap_name, filename):
    fig, ax = plt.subplots(figsize=(6, 6))
    # Display the DP table as a heatmap
    ax.matshow(dp_table, cmap=cmap_name)

    # --- Style the plot for a cleaner look (like the video) ---
    # Remove ticks, labels, and spines
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add visual margin around the grid
    ax.set_xlim(-0.7, cols - 0.3)
    ax.set_ylim(rows - 0.3, -0.7) # Inverted Y for matshow

    # Draw black border around the cells in the minimum path
    for (r, c) in path:
        # Rectangle(xy_lowerleft, width, height, ...)
        rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                             edgecolor='black', facecolor='none', lw=3)
        ax.add_patch(rect)

    # --- Optionally display values inside cells ---
    # for r in range(rows):
    #     for c in range(cols):
    #         ax.text(c, r, f'{dp_table[r, c]}', va='center', ha='center', color='white' if dp_table[r,c] > (dp_table.max()/2) else 'black')


    plt.tight_layout(pad=0.5) # Adjust padding
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Saved visualization to {filename}")
    plt.close()

# ---------- 6. Generate and export example visualization ----------
# Using 'YlOrBr' colormap similar to the video's yellow/brown tones
save_colored_dp_image(dp, path, 'YlOrBr', 'dp_min_path_visualization.png')

# You can uncomment these lines to generate other color versions
# save_colored_dp_image(dp, path, 'Reds', 'dp_red.png')
# save_colored_dp_image(dp, path, 'Blues', 'dp_blue.png')

```

## 🚀 How to Run the Code

1.  **Prerequisites**: Make sure you have Python 3, `numpy`, and `matplotlib` installed.
    ```bash
    pip install numpy matplotlib
    ```
2.  **Save**: Save the code above as a Python file (e.g., `visualize_dp.py`).
3.  **Execute**: Run the script from your terminal.
    ```bash
    python visualize_dp.py
    ```
4.  **Output**: The script will print the calculated DP table, the minimum path sum, the optimal path coordinates, and save a static PNG image (`dp_min_path_visualization.png`) showing the heatmap and the optimal path highlighted.

*(Note: This code generates a static image of the final result. Creating the frame-by-frame animation requires additional steps,  typically involving saving an image for each step of the DP table calculation and then combining them using tools like ffmpeg or imageio.)
