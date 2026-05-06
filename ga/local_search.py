# 2-opt local search

def two_opt_search(route, dist_matrix, max_improvements=50):
    """
    Applies the 2-opt heuristic to refine the route.
    Returns the optimized route.
    """
    best_route = route.copy()
    n = len(best_route)
    improved = True
    iterations = 0
    
    while improved and iterations < max_improvements:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n):
                if j - i == 1: continue 
                
                # Check if reversing the segment from i to j improves the distance
                A = best_route[i - 1]
                B = best_route[i]
                C = best_route[j - 1]
                D = best_route[j % n]

                dist_before = dist_matrix[A][B] + dist_matrix[C][D]
                dist_after = dist_matrix[A][C] + dist_matrix[B][D]
                
                if dist_after < dist_before:
                    # Reverse the segment between i and j-1
                    best_route[i:j] = list(reversed(best_route[i:j]))
                    improved = True
        iterations += 1
    return best_route
