import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refactored TSP solver application.")
    parser.add_argument("--cli", action="store_true", help="Run the non-GUI solver demo.")
    parser.add_argument("--legacy-gui", action="store_true", help="Launch the legacy GUI.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.cli:
        from app.cli import main as cli_main

        cli_main()
        return 0

    if args.legacy_gui:
        from app.ui.legacy_window import main as legacy_main

        legacy_main()
        return 0

    from app.ui.studio_window import launch_gui

    return int(launch_gui())


if __name__ == "__main__":
    raise SystemExit(main())
