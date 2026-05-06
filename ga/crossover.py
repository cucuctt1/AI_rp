import numpy as np
import random



def pmx_crossover(parent1, parent2):
    size = len(parent1)

    cx1, cx2 = sorted(random.sample(range(size), 2))

    child = [-1] * size

    # copy đoạn từ parent1
    child[cx1:cx2] = parent1[cx1:cx2]

    # fill phần còn lại
    for i in range(cx1, cx2):
        val = parent2[i]

        if val not in child:
            pos = i

            while True:
                mapped_val = parent1[pos]
                pos = parent2.index(mapped_val)

                if child[pos] == -1:
                    child[pos] = val
                    break

    # fill chỗ còn lại
    for i in range(size):
        if child[i] == -1:
            child[i] = parent2[i]

    return child

def order_crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [-1] * size
    # Copy segment from parent1
    child[start:end+1] = parent1[start:end+1]
    
    # Fill remaining from parent2 maintaining order
    p2_idx = (end + 1) % size
    c_idx = (end + 1) % size
    count = 0
    while -1 in child:
        if parent2[p2_idx] not in child:
            child[c_idx] = parent2[p2_idx]
            c_idx = (c_idx + 1) % size
        p2_idx = (p2_idx + 1) % size

        count += 1
        if count > 10000:  # Safeguard against infinite loops
            raise RuntimeError("Order Crossover loop detected. Check implementation.")
        
    return child
