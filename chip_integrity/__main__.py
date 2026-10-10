"""Lets ``python -m chip_integrity`` work the same as the ``chip-integrity`` command."""

import sys

from chip_integrity.cli import main

sys.exit(main())
