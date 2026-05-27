import torch


NUM_NODES = 29


# Starter graph for 29 keypoints.
# Replace these edges with your exact landmark topology if you know it.
BASE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (10, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
    (16, 17), (17, 18), (18, 19),
    (10, 20), (20, 21), (21, 22),
    (22, 23), (23, 24),
    (14, 25), (25, 26), (26, 27), (27, 28),
]


def build_adjacency(num_nodes=NUM_NODES, edges=BASE_EDGES):
    adj = torch.eye(num_nodes, dtype=torch.float32)
    for i, j in edges:
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    degree = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
    return adj / degree
