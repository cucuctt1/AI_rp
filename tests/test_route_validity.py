from tsp_ga_app.operators import create_population


def test_created_population_routes_contain_each_city_once():
    population = create_population(pop_size=20, num_cities=8)

    assert len(population) == 20
    for route in population:
        assert sorted(route) == list(range(8))
