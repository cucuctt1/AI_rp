from typing import Any, Dict, List, Optional

import numpy as np


def extract_city_point(item: Any) -> Optional[List[float]]:
    if isinstance(item, dict):
        if "x" in item and "y" in item:
            return [float(item["x"]), float(item["y"])]
        if "coord" in item and isinstance(item["coord"], (list, tuple)) and len(item["coord"]) >= 2:
            return [float(item["coord"][0]), float(item["coord"][1])]
    elif isinstance(item, (list, tuple)) and len(item) >= 2:
        return [float(item[0]), float(item[1])]
    return None


def looks_like_point_map(payload: Dict[Any, Any]) -> bool:
    if not payload:
        return False
    for value in payload.values():
        if extract_city_point(value) is None:
            return False
    return True


def parse_cities_json(payload: Any) -> Optional[np.ndarray]:
    if isinstance(payload, dict):
        if "cities" in payload:
            payload = payload["cities"]
        elif "points" in payload:
            payload = payload["points"]

    points: List[List[float]] = []

    if isinstance(payload, dict):
        def sort_key(item: Any) -> Any:
            key = item[0]
            if isinstance(key, int):
                return (0, key)
            if isinstance(key, str) and key.isdigit():
                return (0, int(key))
            return (1, str(key))

        for _, value in sorted(payload.items(), key=sort_key):
            point = extract_city_point(value)
            if point is not None:
                points.append(point)
    elif isinstance(payload, list):
        for item in payload:
            point = extract_city_point(item)
            if point is not None:
                points.append(point)
    else:
        return None

    if not points:
        return None
    return np.asarray(points, dtype=float)


def extract_city_datasets(payload: Any, fallback_name: str) -> Dict[str, np.ndarray]:
    datasets: Dict[str, np.ndarray] = {}

    if isinstance(payload, dict):
        if "cities" in payload or "points" in payload:
            dataset_name = str(payload.get("name")) if payload.get("name") else fallback_name
            cities = parse_cities_json(payload)
            if cities is not None:
                datasets[dataset_name] = cities
            return datasets

        if looks_like_point_map(payload):
            cities = parse_cities_json(payload)
            if cities is not None:
                datasets[fallback_name] = cities
            return datasets

        for key, value in payload.items():
            if key in {"name", "description"}:
                continue
            cities = parse_cities_json(value)
            if cities is not None:
                datasets[str(key)] = cities
        return datasets

    if isinstance(payload, list):
        cities = parse_cities_json(payload)
        if cities is not None:
            datasets[fallback_name] = cities
    return datasets
