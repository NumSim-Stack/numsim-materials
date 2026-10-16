#!/usr/bin/env python3
"""Entry point of the report generator; see numsim_report/cli.py or --help."""
import sys

from numsim_report.cli import main

if __name__ == "__main__":
    sys.exit(main())
