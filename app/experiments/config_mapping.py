from typing import Any, Dict

from app.config.settings import DEFAULT_CONFIG


def normalize_core_ga_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return config with GUI/app keys mirrored to GAEngine keys."""
    normalized = dict(config or {})

    if "population_size" not in normalized and "pop_size" in normalized:
        normalized["population_size"] = normalized["pop_size"]
    if "pop_size" not in normalized and "population_size" in normalized:
        normalized["pop_size"] = normalized["population_size"]

    if "elitism_k" not in normalized and "elite_size" in normalized:
        normalized["elitism_k"] = normalized["elite_size"]
    if "elite_size" not in normalized and "elitism_k" in normalized:
        normalized["elite_size"] = normalized["elitism_k"]

    normalized.setdefault("population_size", DEFAULT_CONFIG["population_size"])
    normalized.setdefault("pop_size", normalized["population_size"])
    normalized.setdefault("elitism_k", DEFAULT_CONFIG["elitism_k"])
    normalized.setdefault("elite_size", normalized["elitism_k"])
    normalized.setdefault("tournament_size", DEFAULT_CONFIG.get("tournament_size", 3))
    normalized.setdefault("selection_type", DEFAULT_CONFIG["selection_type"])
    normalized.setdefault("crossover_type", DEFAULT_CONFIG["crossover_type"])
    normalized.setdefault("mutation_type", DEFAULT_CONFIG["mutation_type"])
    normalized.setdefault("adaptive_mutation", DEFAULT_CONFIG["adaptive_mutation"])
    normalized.setdefault("local_search_freq", DEFAULT_CONFIG["local_search_freq"])

    return normalized
