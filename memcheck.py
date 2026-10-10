#!/usr/bin/env python3
"""Runs the memory probe (probe 2). The code lives in chip_integrity/memcheck.py; this file keeps `python memcheck.py` working."""

import sys

from chip_integrity.memcheck import main

if __name__ == "__main__":
    sys.exit(main())
