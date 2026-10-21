#!/usr/bin/env python3
"""Runs the full-model probe (probe 4). The code lives in chip_integrity/model.py; this file keeps `python model.py` working."""

import sys

from chip_integrity.model import main

if __name__ == "__main__":
    sys.exit(main())
