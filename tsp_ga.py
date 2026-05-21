import sys

from app.cli import main


if __name__ == "__main__":
    if "--gui" in sys.argv:
        from app.ui.studio_window import launch_gui

        raise SystemExit(launch_gui())
    main()
