#!/usr/bin/env python3
"""Runs the data pattern probe (probe 9). The code lives in chip_integrity/patterns.py; this file keeps `python patterns.py` working."""

import sys

from chip_integrity.patterns import main

if __name__ == "__main__":
    sys.exit(main())
