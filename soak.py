#!/usr/bin/env python3
"""Runs the hours-under-load probe (probe 7). The code lives in chip_integrity/soak.py; this file keeps `python soak.py` working."""

import sys

from chip_integrity.soak import main

if __name__ == "__main__":
    sys.exit(main())
