#!/usr/bin/env python3
"""Runs the transformer kernel probe (probe 3). The code lives in chip_integrity/kernels.py; this file keeps `python kernels.py` working."""

import sys

from chip_integrity.kernels import main

if __name__ == "__main__":
    sys.exit(main())
