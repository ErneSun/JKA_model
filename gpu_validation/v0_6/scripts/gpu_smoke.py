#!/usr/bin/env python3
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    for import_root in (ROOT, ROOT / "src"):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))
    runpy.run_path(str(ROOT / "scripts" / "smoke_v0_6.py"), run_name="__main__")


if __name__ == "__main__":
    main()
