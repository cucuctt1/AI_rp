import random

from ga.mutation import reverse_mutation, scramble_mutation, swap_mutation
from tsp_ga_app.operators import mutation_inversion


def assert_valid_route(route, size):
    assert len(route) == size
    assert sorted(route) == list(range(size))


def test_mutation_outputs_valid_routes():
    base_route = list(range(12))

    random.seed(3)
    assert_valid_route(swap_mutation(base_route.copy(), 1.0), 12)

    random.seed(3)
    assert_valid_route(reverse_mutation(base_route.copy(), 1.0), 12)

    random.seed(3)
    assert_valid_route(scramble_mutation(base_route.copy(), 1.0), 12)

    random.seed(3)
    assert_valid_route(mutation_inversion(base_route.copy(), 1.0), 12)
