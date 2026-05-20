"""Allow `python -m desktop_solver` to launch the GUI."""

from .gui import main

if __name__ == "__main__":
    raise SystemExit(main())
