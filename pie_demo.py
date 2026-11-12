"""Backward-compatible entry point. The code lives in the `pie` package.

Usage:
    python3 pie_demo.py path/to/intrvw23 [--quick]
"""

import sys

from pie.cli import main

if __name__ == "__main__":
    sys.exit(main())
