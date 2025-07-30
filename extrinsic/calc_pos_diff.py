import numpy as np

# Define the three positions
positions = [
    [0.357, 0.155, 1.484],  # Position 1
    [0.358, 0.121, 1.520],  # Position 2
    [0.365, 0.082, 1.560]   # Position 3
]


def calculate_l2_distance(pos1, pos2):
    """Calculate L2 (Euclidean) distance between two 3D points"""
    return np.sqrt(sum((a - b)**2 for a, b in zip(pos1, pos2)))


# Calculate distances between all pairs
print("L2 Distances between positions:")
print("-" * 40)

# Distance between Position 1 and Position 2
dist_1_2 = calculate_l2_distance(positions[0], positions[1])
print(f"Position 1 to Position 2: {dist_1_2:.6f} m")

# Distance between Position 1 and Position 3
dist_1_3 = calculate_l2_distance(positions[0], positions[2])
print(f"Position 1 to Position 3: {dist_1_3:.6f} m")

# Distance between Position 2 and Position 3
dist_2_3 = calculate_l2_distance(positions[1], positions[2])
print(f"Position 2 to Position 3: {dist_2_3:.6f} m")

print("\nAll distances:")
print(f"[1-2]: {dist_1_2:.6f} m")
print(f"[1-3]: {dist_1_3:.6f} m")
print(f"[2-3]: {dist_2_3:.6f} m")

# Alternative using numpy for all pairwise distances
print("\nUsing numpy for verification:")
pos_array = np.array(positions)
for i in range(len(positions)):
    for j in range(i+1, len(positions)):
        dist = np.linalg.norm(pos_array[i] - pos_array[j])
        print(f"Position {i+1} to Position {j+1}: {dist:.6f} m")
