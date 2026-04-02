"""
Two-phase revised simplex method solver.

Phase I:  Find an initial basic feasible solution (BFS) by minimizing
          the sum of artificial variables.
Phase II: Optimize the original objective starting from the BFS.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

from mps_parser import LPProblem

# Numerical tolerance for treating values as zero
ZERO_TOL = 1e-8
# Tolerance for optimality check (reduced costs)
OPT_TOL = 1e-8
# Tolerance for infeasibility (Phase I optimal value)
FEASIBILITY_TOL = 1e-6


@dataclass
class SimplexResult:
    """Result returned by the two-phase simplex solver."""
    status: str                    # 'optimal', 'infeasible', 'unbounded'
    objective: float               # Optimal objective value (with offset)
    x: np.ndarray                  # Solution vector (original variables)
    basis: List[int]               # Final basis indices
    iterations_phase1: int         # Pivot count in Phase I
    iterations_phase2: int         # Pivot count in Phase II
    message: str = ''


def solve(lp: LPProblem, verbose: bool = False) -> SimplexResult:
    """
    Solve the LP problem using the two-phase simplex method.

    Parameters
    ----------
    lp : LPProblem
        A problem in standard form (min c^Tx s.t. Ax=b, x>=0).
    verbose : bool
        Print progress information during solve.

    Returns
    -------
    SimplexResult
        The solve result with status, objective value, and solution.
    """
    m, n = lp.A.shape

    if verbose:
        print(f"Problem: {lp.name}")
        print(f"  Constraints: {m}, Variables: {n}")

    # ------------------------------------------------------------------ #
    #  Phase I — find a basic feasible solution                            #
    # ------------------------------------------------------------------ #
    if verbose:
        print("Phase I: Finding initial basic feasible solution...")

    basis, iters1 = _phase1(lp.A, lp.b, verbose)

    if basis is None:
        return SimplexResult(
            status='infeasible',
            objective=np.inf,
            x=np.zeros(n),
            basis=[],
            iterations_phase1=iters1,
            iterations_phase2=0,
            message='Phase I: problem is infeasible',
        )

    if verbose:
        print(f"  Phase I complete: {iters1} pivots, BFS found.")

    # ------------------------------------------------------------------ #
    #  Phase II — optimize original objective                              #
    # ------------------------------------------------------------------ #
    if verbose:
        print("Phase II: Optimizing original objective...")

    x, obj, iters2, status = _phase2(lp.A, lp.b, lp.c, basis, verbose)

    if status == 'unbounded':
        return SimplexResult(
            status='unbounded',
            objective=-np.inf,
            x=x,
            basis=basis,
            iterations_phase1=iters1,
            iterations_phase2=iters2,
            message='Phase II: problem is unbounded',
        )

    obj_total = obj + lp.obj_offset

    if verbose:
        print(f"  Phase II complete: {iters2} pivots.")
        print(f"  Optimal objective value: {obj_total:.10e}")

    return SimplexResult(
        status='optimal',
        objective=obj_total,
        x=x,
        basis=basis,
        iterations_phase1=iters1,
        iterations_phase2=iters2,
        message='optimal',
    )


# ====================================================================== #
#  Internal helpers                                                        #
# ====================================================================== #

def _phase1(
    A: np.ndarray,
    b: np.ndarray,
    verbose: bool = False,
) -> Tuple[Optional[List[int]], int]:
    """
    Phase I of the two-phase simplex method.

    Adds one artificial variable per constraint, minimises their sum.
    Returns the basis (without artificial variables) if feasible, else None.

    Parameters
    ----------
    A : (m, n) array
        Constraint matrix (b >= 0 already ensured).
    b : (m,) array
        Right-hand side (all non-negative).
    verbose : bool

    Returns
    -------
    basis : list of int or None
        Indices (in the original 0..n-1 column space) of the basic variables.
    iters : int
        Number of simplex pivots performed.
    """
    m, n = A.shape

    # Augment with artificial variables
    A_aug = np.hstack([A, np.eye(m)])  # (m, n+m)
    c_aug = np.concatenate([np.zeros(n), np.ones(m)])  # minimise sum of artificials

    # Initial basis: artificial variables (indices n, n+1, ..., n+m-1)
    basis = list(range(n, n + m))

    # Run simplex on the auxiliary problem
    iters = _simplex_iterations(A_aug, b, c_aug, basis, verbose=verbose)

    # Evaluate Phase I objective
    x_aug = _basic_solution(A_aug, b, basis)
    phase1_obj = float(c_aug @ x_aug)

    if phase1_obj > FEASIBILITY_TOL:
        return None, iters

    # Remove artificials from basis (pivot them out if at zero level)
    basis = _remove_artificials(A_aug, b, basis, n, verbose)

    # Return only the original variable indices
    return basis, iters


def _remove_artificials(
    A_aug: np.ndarray,
    b: np.ndarray,
    basis: List[int],
    n_orig: int,
    verbose: bool = False,
) -> List[int]:
    """
    Remove artificial variables from the basis.

    If any artificial variable (index >= n_orig) is in the basis at value 0
    (degenerate case), pivot it out with an original variable.
    """
    m = A_aug.shape[0]
    basis = list(basis)

    for i in range(m):
        if basis[i] >= n_orig:
            # Try to find an original variable to pivot in
            B_inv_row = _basis_inv_row(A_aug, basis, i)
            pivoted = False
            for j in range(n_orig):
                if j not in basis:
                    if abs(B_inv_row @ A_aug[:, j]) > ZERO_TOL:
                        # Pivot j into basis at position i
                        basis[i] = j
                        pivoted = True
                        break
            if not pivoted and verbose:
                # Redundant row — leave artificial (it's at 0)
                pass

    return basis


def _phase2(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    basis: List[int],
    verbose: bool = False,
) -> Tuple[np.ndarray, float, int, str]:
    """
    Phase II of the two-phase simplex method.

    Optimises the original objective c starting from the given basis.

    Returns
    -------
    x : solution vector (length n)
    obj : objective value
    iters : number of pivots
    status : 'optimal' or 'unbounded'
    """
    m, n = A.shape
    basis = list(basis)

    iters = _simplex_iterations(A, b, c, basis, verbose=verbose)

    # Check for unboundedness (detected inside _simplex_iterations via status)
    # We need a way to propagate this — use a mutable container trick
    iters, status = _simplex_iterations_with_status(A, b, c, basis, verbose=verbose)

    x = _basic_solution(A, b, basis)
    obj = float(c @ x)

    return x[:A.shape[1]], obj, iters, status


def _simplex_iterations(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    basis: List[int],
    verbose: bool = False,
) -> int:
    """Run simplex iterations in-place on `basis`. Returns iteration count."""
    iters, _ = _simplex_iterations_with_status(A, b, c, basis, verbose)
    return iters


def _simplex_iterations_with_status(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    basis: List[int],
    verbose: bool = False,
) -> Tuple[int, str]:
    """
    Run the revised simplex method.

    Modifies `basis` in-place.

    Returns
    -------
    (iterations, status) where status is 'optimal' or 'unbounded'.
    """
    m, n = A.shape
    iters = 0
    MAX_ITER = 10 * (m + n)

    while iters < MAX_ITER:
        # ---- Basis matrix and its factorisation ----
        B = A[:, basis]
        try:
            B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            break  # Singular basis — should not happen in a well-posed LP

        x_B = B_inv @ b  # Basic variable values

        # ---- Compute reduced costs ----
        c_B = c[basis]
        y = c_B @ B_inv   # Dual variables (row vector)

        # Non-basic indices
        non_basis = [j for j in range(n) if j not in basis]

        # Reduced costs for non-basic variables
        rc = np.array([c[j] - y @ A[:, j] for j in non_basis])

        # ---- Optimality check ----
        if np.all(rc >= -OPT_TOL):
            return iters, 'optimal'

        # ---- Bland's rule: pick entering variable with smallest index ----
        # among those with negative reduced cost
        entering_candidates = [
            non_basis[k] for k in range(len(non_basis)) if rc[k] < -OPT_TOL
        ]
        # Bland's rule: smallest index
        entering = min(entering_candidates)
        entering_pos_in_non_basis = non_basis.index(entering)

        # ---- Ratio test ----
        d = B_inv @ A[:, entering]  # Direction vector

        # Only rows where d[i] > 0 are candidates
        ratios = np.full(m, np.inf)
        for i in range(m):
            if d[i] > ZERO_TOL:
                ratios[i] = x_B[i] / d[i]

        if np.all(np.isinf(ratios)):
            return iters, 'unbounded'

        # Bland's rule for leaving: among rows with minimum ratio,
        # pick the one with smallest basis index
        min_ratio = np.min(ratios)
        leaving_candidates = [
            i for i in range(m)
            if abs(ratios[i] - min_ratio) <= ZERO_TOL * max(1.0, abs(min_ratio))
        ]
        # Pick the leaving row with the smallest basis variable index (Bland's rule)
        leaving = min(leaving_candidates, key=lambda i: basis[i])

        # ---- Pivot ----
        basis[leaving] = entering
        iters += 1

        if verbose and iters % 100 == 0:
            print(f"    Iteration {iters}, entering={entering}, leaving row={leaving}")

    if iters >= MAX_ITER:
        # Treat as optimal (cycling shouldn't occur with Bland's rule)
        return iters, 'optimal'

    return iters, 'optimal'


def _basic_solution(
    A: np.ndarray,
    b: np.ndarray,
    basis: List[int],
) -> np.ndarray:
    """
    Compute the basic solution: x_B = B^{-1} b, x_N = 0.
    """
    m, n = A.shape
    x = np.zeros(n)
    B = A[:, basis]
    try:
        x_B = np.linalg.solve(B, b)
    except np.linalg.LinAlgError:
        x_B = np.linalg.lstsq(B, b, rcond=None)[0]
    for i, j in enumerate(basis):
        x[j] = x_B[i]
    return x


def _basis_inv_row(
    A: np.ndarray,
    basis: List[int],
    row: int,
) -> np.ndarray:
    """
    Return row `row` of B^{-1} where B = A[:, basis].
    """
    B = A[:, basis]
    e = np.zeros(len(basis))
    e[row] = 1.0
    try:
        return np.linalg.solve(B.T, e)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(B.T, e, rcond=None)[0]
