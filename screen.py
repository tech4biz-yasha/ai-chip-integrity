#!/usr/bin/env python3
"""Runs the compute probe (probe 1). The code lives in chip_integrity/screen.py; this file keeps `python screen.py` working."""

import sys

from chip_integrity.screen import main

if __name__ == "__main__":
    sys.exit(main())
