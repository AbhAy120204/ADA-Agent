"""
CLI entry point for the Autonomous Data Analyst Agent.

Usage:
    python main.py --file data/examples/sales.csv
    python main.py --file data/examples/sales.csv --iterations 3
"""

import argparse
import sys
from pathlib import Path

from agent.graph import run_analysis


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Data Analyst Agent — give it a CSV, it finds insights."
    )
    parser.add_argument(
        "--file", required=True, help="Path to the CSV file to analyze"
    )
    parser.add_argument(
        "--iterations", type=int, default=5,
        help="Max number of analysis loops (default: 5)"
    )
    args = parser.parse_args()

    csv_path = Path(args.file)
    if not csv_path.exists():
        print(f"Error: File not found — {csv_path}")
        sys.exit(1)

    print(f"\nAnalyzing: {csv_path}")
    print(f"Max iterations: {args.iterations}")
    print("=" * 60)

    result = run_analysis(str(csv_path), max_iterations=args.iterations)

    print("\n\nAll insights collected:")
    for i, insight in enumerate(result["insights"], 1):
        print(f"  {i}. {insight}")


if __name__ == "__main__":
    main()
