#!/usr/bin/env python3
"""Runs the checksum-protected matrix multiply (probe 6). The code lives in chip_integrity/abft.py; this file keeps `python abft.py` working."""

import sys

from chip_integrity.abft import main

if __name__ == "__main__":
    sys.exit(main())
