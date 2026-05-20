import random

from ga.crossover import order_crossover, pmx_crossover
from tsp_ga_app.operators import crossover_OX1


def assert_valid_route(route, size):
    assert len(route) == size
    assert sorted(route) == list(range(size))


def test_crossover_outputs_valid_routes():
    parent_a = list(range(10))
    parent_b = list(reversed(parent_a))

    random.seed(7)
    assert_valid_route(pmx_crossover(parent_a, parent_b), 10)

    random.seed(7)
    assert_valid_route(order_crossover(parent_a, parent_b), 10)

    random.seed(7)
    assert_valid_route(crossover_OX1(parent_a, parent_b, crossover_rate=1.0), 10)
