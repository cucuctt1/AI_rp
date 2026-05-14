"""
tsp_csv_to_json.py
------------------
Converts the TSPLIB CSV dataset into one JSON file per TSP instance.
Takes the top 50 instances (by first appearance order in the CSV).

CSV columns:
  instance_id       -> used as the JSON "name"
  num_cities        -> number of cities
  city_coordinates  -> JSON string: [[x0,y0], [x1,y1], ...]
  distance_matrix   -> (ignored)
  best_route        -> best known route order
  total_distance    -> best known total distance

Usage:
  python tsp_csv_to_json.py <input.csv> [output_dir]

  output_dir defaults to ./tsp_json_output/
"""

import csv
import json
import sys
from pathlib import Path

TOP_N = 50

def set_field_size_limit():
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int //= 2

set_field_size_limit()


def slugify(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name)).strip("_")


def convert(csv_path: str, output_dir: str) -> None:
    csv_path   = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        sys.exit(f"[ERROR] File not found: {csv_path}")

    exported = 0

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if exported >= TOP_N:
                break

            instance_id     = row.get("instance_id", f"instance_{exported}").strip()
            num_cities      = row.get("num_cities", "?").strip()
            coords_raw      = row.get("city_coordinates", "").strip()
            best_route      = row.get("best_route", "").strip()
            total_distance  = row.get("total_distance", "").strip()

            # Parse [[x0,y0], [x1,y1], ...]
            try:
                coords = json.loads(coords_raw)
            except json.JSONDecodeError as e:
                print(f"  [WARN] Skipping '{instance_id}': bad city_coordinates — {e}")
                continue

            cities = []
            for i, point in enumerate(coords):
                try:
                    cities.append({"id": i, "x": float(point[0]), "y": float(point[1])})
                except (IndexError, TypeError, ValueError) as e:
                    print(f"  [WARN] Bad point at index {i} in '{instance_id}': {e}")

            # Round total_distance for display
            try:
                dist_display = round(float(total_distance), 4)
            except (ValueError, TypeError):
                dist_display = "N/A"

            payload = {
                "name": instance_id,
                "description": (
                    f"TSP instance '{instance_id}' | "
                    f"Cities: {num_cities} | "
                    f"Best known distance: {dist_display} | "
                    f"Best route: {best_route}"
                ),
                "cities": cities,
            }

            out_file = output_dir / f"{slugify(instance_id)}.json"
            with open(out_file, "w", encoding="utf-8") as out:
                json.dump(payload, out, indent=2)

            exported += 1
            print(f"  [{exported:>3}] {instance_id:20s} -> {out_file.name}  ({len(cities)} cities, dist={dist_display})")

    print(f"\nDone! {exported} JSON files written to: {output_dir}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    csv_input  = sys.argv[1]
    out_folder = sys.argv[2] if len(sys.argv) > 2 else "tsp_json_output"
    convert(csv_input, out_folder)