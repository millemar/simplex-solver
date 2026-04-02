#!/usr/bin/env python3
"""
Main entry point for the two-phase simplex solver.

Usage
-----
    python main.py afiro
    python main.py data/afiro.mps
    python main.py --download-all
"""

import argparse
import sys
import time

import numpy as np
from mps_parser import parse_mps
from simplex import solve
from netlib_downloader import get_problem_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Two-phase simplex solver for Netlib LP test problems.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "problem",
        nargs="?",
        help="MPS file path or Netlib problem name (e.g. 'afiro')",
    )
    parser.add_argument(
        "--download-all",
        action="store_true",
        help="Download all standard Netlib test problems to data/",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed progress during solve",
    )

    args = parser.parse_args(argv)

    if args.download_all:
        from netlib_downloader import download_all
        print("Downloading all standard Netlib problems...")
        download_all()
        return 0

    if not args.problem:
        parser.print_help()
        return 1

    # ------------------------------------------------------------------ #
    #  Resolve file path                                                   #
    # ------------------------------------------------------------------ #
    try:
        mps_path = get_problem_path(args.problem)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------ #
    #  Parse MPS file                                                      #
    # ------------------------------------------------------------------ #
    print(f"\nParsing: {mps_path}")
    try:
        lp = parse_mps(mps_path)
    except Exception as e:
        print(f"Parse error: {e}", file=sys.stderr)
        return 1

    m, n = lp.A.shape
    nnz = int(np.count_nonzero(lp.A))
    print(f"Problem '{lp.name}': {m} rows, {n} cols, {nnz} non-zeros")

    # ------------------------------------------------------------------ #
    #  Solve                                                               #
    # ------------------------------------------------------------------ #
    t0 = time.time()
    result = solve(lp, verbose=args.verbose)
    elapsed = time.time() - t0

    # ------------------------------------------------------------------ #
    #  Report results                                                      #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print(f"Status:  {result.status}")
    print(f"Elapsed: {elapsed:.3f}s")
    print(f"Phase I iterations:  {result.iterations_phase1}")
    print(f"Phase II iterations: {result.iterations_phase2}")

    if result.status == 'optimal':
        print(f"Optimal objective value: {result.objective:.10e}")
        # Print non-zero solution variables (original vars only)
        print("\nNon-zero solution variables (original):")
        n_orig = lp.n_orig
        printed = 0
        for j in range(min(n_orig, len(result.x))):
            if abs(result.x[j]) > 1e-10:
                print(f"  {lp.var_names[j]:20s} = {result.x[j]:.10e}")
                printed += 1
        if printed == 0:
            print("  (all zero)")
    elif result.status == 'infeasible':
        print("Problem is INFEASIBLE.")
    elif result.status == 'unbounded':
        print("Problem is UNBOUNDED.")
    else:
        print(f"Message: {result.message}")

    print('='*60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
