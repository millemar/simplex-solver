"""
MPS file parser for the Netlib LP test library.

Parses standard fixed-format MPS files and converts them into standard form:
    minimize c^T x  subject to  Ax = b, x >= 0
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class LPProblem:
    """Represents a linear programming problem in standard form."""
    name: str
    # Standard form: min c^T x  s.t.  A x = b, x >= 0
    c: np.ndarray          # Objective coefficients (n,)
    A: np.ndarray          # Constraint matrix (m, n)
    b: np.ndarray          # Right-hand side (m,)
    var_names: List[str]   # Variable names (length n)
    row_names: List[str]   # Constraint row names (length m)
    n_orig: int            # Number of original (non-slack) variables
    obj_offset: float = 0.0  # Constant offset in objective


def parse_mps(filepath: str) -> LPProblem:
    """
    Parse an MPS file and return an LPProblem in standard form.

    Parameters
    ----------
    filepath : str
        Path to the MPS file (plain text or .gz compressed).

    Returns
    -------
    LPProblem
        The parsed LP in standard form (min c^T x, Ax = b, x >= 0).
    """
    import gzip

    # Read lines
    if filepath.endswith('.gz'):
        with gzip.open(filepath, 'rt') as f:
            lines = f.readlines()
    else:
        with open(filepath, 'r') as f:
            lines = f.readlines()

    # ------------------------------------------------------------------ #
    #  Raw data containers                                                 #
    # ------------------------------------------------------------------ #
    problem_name = ''
    row_types: Dict[str, str] = {}   # row_name -> 'N'/'L'/'G'/'E'
    row_order: List[str] = []        # order of non-objective rows
    obj_name: Optional[str] = None

    # column_data[var_name] = {row_name: coeff}
    column_data: Dict[str, Dict[str, float]] = {}
    col_order: List[str] = []        # insertion order of variables

    # rhs_data[row_name] = value
    rhs_data: Dict[str, float] = {}

    # ranges_data[row_name] = range value
    ranges_data: Dict[str, float] = {}

    # bounds_data[var_name] = list of (bound_type, value)
    bounds_data: Dict[str, List[Tuple[str, float]]] = {}

    section = None
    in_integer_block = False

    for raw_line in lines:
        line = raw_line.rstrip('\n')

        # Skip blank lines and comments ($ at start)
        stripped = line.strip()
        if not stripped or stripped.startswith('$'):
            continue

        # Section header lines start in column 1 (not indented)
        if line[0] != ' ' and line[0] != '\t':
            keyword = stripped.split()[0].upper()
            if keyword == 'NAME':
                parts = stripped.split()
                problem_name = parts[1] if len(parts) > 1 else ''
                section = 'NAME'
            elif keyword == 'ROWS':
                section = 'ROWS'
            elif keyword == 'COLUMNS':
                section = 'COLUMNS'
                in_integer_block = False
            elif keyword == 'RHS':
                section = 'RHS'
            elif keyword == 'RANGES':
                section = 'RANGES'
            elif keyword == 'BOUNDS':
                section = 'BOUNDS'
            elif keyword == 'ENDATA':
                break
            else:
                section = keyword
            continue

        # Data lines
        if section == 'ROWS':
            parts = stripped.split()
            if len(parts) >= 2:
                rtype, rname = parts[0].upper(), parts[1]
                row_types[rname] = rtype
                if rtype == 'N' and obj_name is None:
                    obj_name = rname
                elif rtype != 'N':
                    row_order.append(rname)

        elif section == 'COLUMNS':
            # Detect integer marker lines
            if "'MARKER'" in line or "MARKER" in stripped:
                if "'INTORG'" in line or 'INTORG' in stripped:
                    in_integer_block = True
                elif "'INTEND'" in line or 'INTEND' in stripped:
                    in_integer_block = False
                continue

            # Parse: col_name  row_name  value  [row_name  value]
            parts = stripped.split()
            if len(parts) < 3:
                continue
            col_name = parts[0]
            if col_name not in column_data:
                column_data[col_name] = {}
                col_order.append(col_name)
            # First (row, value) pair
            try:
                column_data[col_name][parts[1]] = float(parts[2])
            except (ValueError, IndexError):
                pass
            # Optional second (row, value) pair
            if len(parts) >= 5:
                try:
                    column_data[col_name][parts[3]] = float(parts[4])
                except (ValueError, IndexError):
                    pass

        elif section == 'RHS':
            parts = stripped.split()
            if len(parts) < 3:
                continue
            # rhs_name  row_name  value  [row_name  value]
            try:
                rhs_data[parts[1]] = float(parts[2])
            except (ValueError, IndexError):
                pass
            if len(parts) >= 5:
                try:
                    rhs_data[parts[3]] = float(parts[4])
                except (ValueError, IndexError):
                    pass

        elif section == 'RANGES':
            parts = stripped.split()
            if len(parts) < 3:
                continue
            try:
                ranges_data[parts[1]] = float(parts[2])
            except (ValueError, IndexError):
                pass
            if len(parts) >= 5:
                try:
                    ranges_data[parts[3]] = float(parts[4])
                except (ValueError, IndexError):
                    pass

        elif section == 'BOUNDS':
            parts = stripped.split()
            if len(parts) < 3:
                continue
            btype = parts[0].upper()
            # parts[1] is bound set name, parts[2] is var name
            if len(parts) < 3:
                continue
            var_name = parts[2]
            if var_name not in bounds_data:
                bounds_data[var_name] = []
            if btype in ('FR', 'MI', 'PL', 'BV'):
                bounds_data[var_name].append((btype, 0.0))
            elif len(parts) >= 4:
                try:
                    val = float(parts[3])
                    bounds_data[var_name].append((btype, val))
                except (ValueError, IndexError):
                    pass
            else:
                # LO/UP/FX without value — treat as 0
                bounds_data[var_name].append((btype, 0.0))

    # ------------------------------------------------------------------ #
    #  Build standard-form LP                                              #
    # ------------------------------------------------------------------ #
    m_orig = len(row_order)
    n_orig = len(col_order)

    # Determine variable bounds: lower bound lb, upper bound ub (may be +inf)
    lb = np.zeros(n_orig)
    ub = np.full(n_orig, np.inf)

    for j, cname in enumerate(col_order):
        if cname in bounds_data:
            for btype, val in bounds_data[cname]:
                if btype == 'LO':
                    lb[j] = val
                elif btype == 'UP':
                    ub[j] = val
                elif btype == 'FX':
                    lb[j] = val
                    ub[j] = val
                elif btype == 'FR':
                    lb[j] = -np.inf
                    ub[j] = np.inf
                elif btype == 'MI':
                    lb[j] = -np.inf
                elif btype == 'PL':
                    lb[j] = 0.0
                    ub[j] = np.inf
                elif btype == 'BV':
                    lb[j] = 0.0
                    ub[j] = 1.0

    # Build raw A_orig (m_orig x n_orig) and b_orig, c_orig
    A_orig = np.zeros((m_orig, n_orig))
    b_orig = np.zeros(m_orig)
    c_orig = np.zeros(n_orig)
    obj_offset = 0.0

    for j, cname in enumerate(col_order):
        coeffs = column_data[cname]
        if obj_name and obj_name in coeffs:
            c_orig[j] = coeffs[obj_name]
        for i, rname in enumerate(row_order):
            if rname in coeffs:
                A_orig[i, j] = coeffs[rname]

    for i, rname in enumerate(row_order):
        b_orig[i] = rhs_data.get(rname, 0.0)

    # ------------------------------------------------------------------ #
    #  Transform bounded variables                                         #
    # ------------------------------------------------------------------ #
    # For each variable with finite lower bound != 0: substitute x_j = x_j' + lb_j
    # For each variable with finite upper bound: add upper bound constraint
    # For free variables (lb = -inf): split into x+ - x-

    # We work with shifted variables: x' = x - lb  where lb is finite
    # Shift RHS and objective constant
    for j in range(n_orig):
        if np.isfinite(lb[j]) and lb[j] != 0.0:
            # b -= A_orig[:,j] * lb[j]
            b_orig -= A_orig[:, j] * lb[j]
            obj_offset += c_orig[j] * lb[j]
            ub[j] -= lb[j]
            lb[j] = 0.0

    # Handle free variables: split x_j (lb=-inf) into x_j_pos - x_j_neg
    free_vars = [j for j in range(n_orig) if not np.isfinite(lb[j])]
    # For simplicity: shift MI variables so lb[j] = 0 by a big-M approach
    # (proper treatment: split free variable into two non-negatives)
    # We'll do the split approach

    extra_cols_A: List[np.ndarray] = []
    extra_c: List[float] = []
    extra_var_names: List[str] = []
    extra_ub: List[float] = []

    # Replace free variables with x_j = x_j_pos - x_j_neg, both >= 0
    # We'll zero out the original column and replace with two new columns
    replaced_free: Dict[int, Tuple[int, int]] = {}  # j -> (pos_idx, neg_idx)
    next_idx = n_orig + len(extra_cols_A)

    for j in free_vars:
        pos_col = A_orig[:, j].copy()
        neg_col = -A_orig[:, j].copy()
        pos_c = c_orig[j]
        neg_c = -c_orig[j]
        extra_cols_A.append(pos_col)
        extra_c.append(pos_c)
        extra_var_names.append(col_order[j] + '_pos')
        extra_ub.append(np.inf)
        extra_cols_A.append(neg_col)
        extra_c.append(neg_c)
        extra_var_names.append(col_order[j] + '_neg')
        extra_ub.append(np.inf)
        # Zero out original free column (will be excluded)
        A_orig[:, j] = 0.0
        c_orig[j] = 0.0

    # Build augmented matrix (including free variable replacements)
    if extra_cols_A:
        A_extra = np.column_stack(extra_cols_A)
        A_aug = np.hstack([A_orig, A_extra])
        c_aug = np.concatenate([c_orig, extra_c])
        lb_aug = np.concatenate([lb, np.zeros(len(extra_c))])
        ub_aug = np.concatenate([ub, extra_ub])
        var_names_aug = col_order + extra_var_names
    else:
        A_aug = A_orig
        c_aug = c_orig
        lb_aug = lb
        ub_aug = ub
        var_names_aug = list(col_order)

    # Remove original free variable columns (they were zeroed out)
    keep_cols = [j for j in range(len(var_names_aug))
                 if j not in free_vars or j >= n_orig]
    # Actually free_vars indices are in 0..n_orig-1, extra cols start at n_orig
    keep_cols = [j for j in range(len(var_names_aug)) if j not in free_vars]
    A_aug = A_aug[:, keep_cols]
    c_aug = c_aug[keep_cols]
    lb_aug = lb_aug[keep_cols]
    ub_aug = ub_aug[keep_cols]
    var_names_aug = [var_names_aug[j] for j in keep_cols]

    n_aug = len(var_names_aug)

    # ------------------------------------------------------------------ #
    #  Add slack/surplus variables for inequality constraints              #
    # ------------------------------------------------------------------ #
    # row_types: L -> add slack s >= 0, so Ax + s = b
    #            G -> subtract surplus s >= 0, so Ax - s = b
    #            E -> already equality

    slack_cols: List[np.ndarray] = []
    slack_c: List[float] = []
    slack_names: List[str] = []
    slack_ub: List[float] = []

    for i, rname in enumerate(row_order):
        rtype = row_types[rname]
        if rtype == 'L':
            col = np.zeros(m_orig)
            col[i] = 1.0
            slack_cols.append(col)
            slack_c.append(0.0)
            slack_names.append('_slack_' + rname)
            slack_ub.append(np.inf)
        elif rtype == 'G':
            col = np.zeros(m_orig)
            col[i] = -1.0
            slack_cols.append(col)
            slack_c.append(0.0)
            slack_names.append('_surplus_' + rname)
            slack_ub.append(np.inf)

    if slack_cols:
        A_slk = np.column_stack(slack_cols)
        A_full = np.hstack([A_aug, A_slk])
        c_full = np.concatenate([c_aug, slack_c])
        lb_full = np.concatenate([lb_aug, np.zeros(len(slack_c))])
        ub_full = np.concatenate([ub_aug, slack_ub])
        var_names_full = var_names_aug + slack_names
    else:
        A_full = A_aug
        c_full = c_aug
        lb_full = lb_aug
        ub_full = ub_aug
        var_names_full = var_names_aug

    n_full = len(var_names_full)

    # ------------------------------------------------------------------ #
    #  Handle RANGES (double-sided constraints)                            #
    # ------------------------------------------------------------------ #
    # For a range r on row i:
    #   L row: b[i] - |r| <= Ax <= b[i]  (add range slack)
    #   G row: b[i] <= Ax <= b[i] + |r|
    #   E row: b[i] <= Ax <= b[i] + |r| if r > 0, else b[i]+r <= Ax <= b[i]
    # For simplicity we'll add range slacks as additional variables/constraints
    # (not fully implemented here; ranges are uncommon in basic test problems)

    b_full = b_orig.copy()

    # ------------------------------------------------------------------ #
    #  Ensure b >= 0 (flip rows if needed)                                 #
    # ------------------------------------------------------------------ #
    for i in range(m_orig):
        if b_full[i] < 0:
            A_full[i, :] *= -1
            b_full[i] *= -1

    # ------------------------------------------------------------------ #
    #  Handle finite upper bounds: add ub constraint x_j + s = ub_j       #
    # ------------------------------------------------------------------ #
    bounded_rows: List[Tuple[int, float, str]] = []  # (col_idx, ub, var_name)
    for j, vname in enumerate(var_names_full):
        if np.isfinite(ub_full[j]) and ub_full[j] < 1e30:
            bounded_rows.append((j, ub_full[j], vname))

    if bounded_rows:
        n_bnd = len(bounded_rows)
        m_bnd = m_orig + n_bnd
        A_bnd = np.zeros((m_bnd, n_full + n_bnd))
        A_bnd[:m_orig, :n_full] = A_full
        b_bnd = np.zeros(m_bnd)
        b_bnd[:m_orig] = b_full
        c_bnd = np.concatenate([c_full, np.zeros(n_bnd)])
        var_names_bnd = list(var_names_full)
        row_names_bnd = list(row_order)

        for k, (j, ubval, vname) in enumerate(bounded_rows):
            row_idx = m_orig + k
            A_bnd[row_idx, j] = 1.0
            A_bnd[row_idx, n_full + k] = 1.0
            b_bnd[row_idx] = ubval
            var_names_bnd.append('_ubs_' + vname)
            row_names_bnd.append('_ubrow_' + vname)

        A_final = A_bnd
        b_final = b_bnd
        c_final = c_bnd
        var_names_final = var_names_bnd
        row_names_final = row_names_bnd
    else:
        A_final = A_full
        b_final = b_full
        c_final = c_full
        var_names_final = list(var_names_full)
        row_names_final = list(row_order)

    return LPProblem(
        name=problem_name,
        c=c_final,
        A=A_final,
        b=b_final,
        var_names=var_names_final,
        row_names=row_names_final,
        n_orig=n_orig,
        obj_offset=obj_offset,
    )
