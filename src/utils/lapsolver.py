import numpy as np

def solve_dense(cost_matrix):
    """
    Solve the minimum cost bipartite matching problem for a dense cost matrix.

    Parameters:
    cost_matrix (np.ndarray): A 2D array representing the cost matrix.

    Returns:
    tuple: Two lists representing row indices and column indices of the matching.
    """
    if cost_matrix.ndim != 2:
        raise ValueError("Input matrix must be 2-dimensional.")

    nrows, ncols = cost_matrix.shape

    if nrows == 0 or ncols == 0:
        return [], []

    # Flatten and calculate maximum finite cost
    data = cost_matrix.flatten()
    finite_mask = np.isfinite(data)
    if not np.any(finite_mask):
        return [], []

    max_abs_cost = np.max(np.abs(data[finite_mask]))
    r = min(nrows, ncols)
    n = max(nrows, ncols)
    LARGE_COST = 2 * r * max_abs_cost + 1

    # Create extended cost matrix
    extended_costs = np.full((n, n), LARGE_COST)
    extended_costs[:nrows, :ncols] = np.where(np.isfinite(cost_matrix), cost_matrix, LARGE_COST)

    # Solve using the dense algorithm
    row_match = [-1] * n
    col_match = [-1] * n

    solve_dense_core(extended_costs, row_match, col_match)

    # Extract results for the original size
    row_indices = []
    col_indices = []

    for i in range(nrows):
        mate = row_match[i]
        if mate < ncols and extended_costs[i, mate] != LARGE_COST:
            row_indices.append(i)
            col_indices.append(mate)

    return row_indices, col_indices


def solve_dense_core(costs, row_match, col_match):
    """
    Core dense solver logic based on the augmenting path algorithm.
    Refer to `dense.hpp` for algorithm details.

    Parameters:
    costs (np.ndarray): Extended cost matrix.
    row_match (list): List for storing row matches.
    col_match (list): List for storing column matches.
    """
    n = costs.shape[0]
    u = np.zeros(n)
    v = np.zeros(n)

    # Construct dual feasible solution
    for i in range(n):
        u[i] = np.min(costs[i])

    for j in range(n):
        v[j] = np.min(costs[:, j] - u)

    # Construct primal solution
    for i in range(n):
        for j in range(n):
            if col_match[j] != -1:
                continue
            if abs(costs[i, j] - u[i] - v[j]) < 1e-10:
                row_match[i] = j
                col_match[j] = i
                break

    mated = sum(1 for x in row_match if x != -1)
    dist = np.zeros(n)
    dad = [-1] * n
    seen = np.zeros(n, dtype=bool)

    while mated < n:
        s = next(i for i, x in enumerate(row_match) if x == -1)

        dist[:] = costs[s] - u[s] - v
        seen[:] = False
        dad[:] = [-1] * len(dad)


        while True:
            j = np.argmin(np.where(seen, np.inf, dist))
            seen[j] = True

            if col_match[j] == -1:
                break

            i = col_match[j]
            for k in range(n):
                if seen[k]:
                    continue
                new_dist = dist[j] + costs[i, k] - u[i] - v[k]
                if dist[k] > new_dist:
                    dist[k] = new_dist
                    dad[k] = j

        # Update dual variables
        for k in range(n):
            if not seen[k] or k == j:
                continue
            i = col_match[k]
            v[k] += dist[k] - dist[j]
            u[i] -= dist[k] - dist[j]
        u[s] += dist[j]

        # Augment along the path
        while dad[j] >= 0:
            d = dad[j]
            col_match[j] = col_match[d]
            row_match[col_match[j]] = j
            j = d

        col_match[j] = s
        row_match[s] = j
        mated += 1