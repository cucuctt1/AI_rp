import sys

from tsp_ga_app.main import main


if __name__ == "__main__":
    if "--gui" in sys.argv:
        from tsp_ga_app.gui import launch_gui

        raise SystemExit(launch_gui())
    main()
