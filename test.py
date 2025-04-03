import numpy as np
import matplotlib.pyplot as plt

# ---------- 1. Define the grid ----------
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
for i in range(1, rows):
    dp[i, 0] = dp[i - 1, 0] + grid[i, 0]
for j in range(1, cols):
    dp[0, j] = dp[0, j - 1] + grid[0, j]

# ---------- 3. Fill the DP table ----------
for i in range(1, rows):
    for j in range(1, cols):
        dp[i, j] = min(dp[i - 1, j], dp[i, j - 1]) + grid[i, j]

# ---------- 4. Backtrack the minimum path ----------
path = [(rows - 1, cols - 1)]
i, j = rows - 1, cols - 1
while i > 0 or j > 0:
    if i == 0:
        j -= 1
    elif j == 0:
        i -= 1
    elif dp[i - 1, j] < dp[i, j - 1]:
        i -= 1
    else:
        j -= 1
    path.append((i, j))
path.reverse()

# ---------- 5. Function to visualize and save the result ----------
def save_colored_dp_image(cmap_name, filename):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.matshow(dp, cmap=cmap_name)

    # Remove ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(left=False, bottom=False)

    # Add visual margin around the grid
    ax.set_xlim(-0.7, cols - 0.3)
    ax.set_ylim(rows - 0.3, -0.7)

    # Draw black border around the minimum path
    for (i, j) in path:
        rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, edgecolor='black', facecolor='none', lw=3)
        ax.add_patch(rect)

    plt.box(False)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# ---------- 6. Generate and export red, yellow, and blue versions ----------
save_colored_dp_image('Reds', 'dp_red.png')
save_colored_dp_image('YlOrBr', 'dp_yellow.png')
save_colored_dp_image('Blues', 'dp_blue.png')

