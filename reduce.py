#!/usr/bin/env python3
"""Runs the reductions probe (probe 5). The code lives in chip_integrity/reduce.py; this file keeps `python reduce.py` working."""

import sys

from chip_integrity.reduce import main

if __name__ == "__main__":
    sys.exit(main())
