# simplex-solver

A two-phase simplex method solver in Python that can parse and solve linear programming problems from the [Netlib LP test library](https://netlib.org/lp/data/) (MPS format).

## Features

- **MPS parser** — reads standard fixed-format MPS files (ROWS, COLUMNS, RHS, BOUNDS, RANGES sections)
- **Phase I** — finds an initial basic feasible solution (BFS) using artificial variables
- **Phase II** — optimises the objective using the revised simplex method with Bland's rule for anti-cycling
- **Netlib downloader** — automatically fetches and caches `.mps.gz` test problems
- **CLI** — solve any Netlib problem or local MPS file from the command line

## Project Structure

```
simplex-solver/
├── README.md              # This file
├── requirements.txt       # numpy, pytest
├── main.py                # CLI entry point
├── mps_parser.py          # MPS file parser → standard form matrices
├── simplex.py             # Two-phase simplex solver
├── netlib_downloader.py   # Download & cache Netlib LP problems
├── tests/
│   ├── __init__.py
│   └── test_solver.py     # Tests against known Netlib optimal values
└── data/                  # Cached MPS files (git-ignored)
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Solve by Netlib problem name (downloaded automatically)
python main.py afiro
python main.py adlittle

# Solve a local MPS file
python main.py data/afiro.mps

# Verbose output (shows iteration counts)
python main.py -v sc50b

# Download all standard test problems
python main.py --download-all
```

### Programmatic API

```python
import numpy as np
from simplex import solve_linear_program

# From an MPS data file
result = solve_linear_program(data_file="data/afiro.mps")

# Or directly from standard-form arrays (Ax=b, x>=0)
A = np.array([[1, 1, 1, 0], [1, -1, 0, 1]], dtype=float)
b = np.array([4.0, 2.0])
c = np.array([-1.0, -2.0, 0.0, 0.0])
result = solve_linear_program(A=A, b=b, c=c)
```

### Example Output

```
Parsing: data/afiro.mps
Problem 'AFIRO': 27 rows, 51 cols, 102 non-zeros

============================================================
Status:  optimal
Elapsed: 0.012s
Phase I iterations:  8
Phase II iterations: 15
Optimal objective value: -4.6475314286e+02

Non-zero solution variables (original):
  X01                   = 8.0000000000e+01
  X02                   = 5.5000000000e+01
  ...
============================================================
```

## Algorithm

### MPS Parsing (`mps_parser.py`)

Converts MPS-format LP problems to standard form:

> Minimise **c**ᵀ**x**  subject to  **Ax** = **b**,  **x** ≥ 0

- `L` (≤) rows → add slack variable
- `G` (≥) rows → subtract surplus variable
- `E` rows → already equality
- Variable bounds (`LO`, `UP`, `FX`, `FR`, `MI`, `PL`, `BV`) handled by shifting/splitting

### Phase I — Find Initial Vertex

1. Add one artificial variable per constraint
2. Minimise the sum of artificial variables (auxiliary LP)
3. If optimal value > 0 → original problem is **infeasible**
4. If optimal value = 0 → extract the BFS; pivot out any remaining artificials (degenerate case)

### Phase II — Revised Simplex

Starting from the BFS found in Phase I:

1. Compute reduced costs: **c**_N − **c**_B **B**⁻¹ **N**
2. **Pricing**: enter variable with most-negative reduced cost (Bland's rule for anti-cycling)
3. **Ratio test**: minimum ratio test to find leaving variable
4. **Pivot**: update basis
5. Repeat until all reduced costs ≥ 0 (optimal) or unbounded detected

## Tests

Run the test suite (downloads test problems automatically):

```bash
pytest tests/
```

Known optimal values validated:

| Problem  | Optimal Value       |
|----------|---------------------|
| afiro    | −4.6475314286×10²  |
| adlittle |  2.2549496316×10⁵  |
| sc50a    | −6.4575077059×10¹  |
| sc50b    | −7.0000000000×10¹  |
| kb2      | −1.7499001299×10³  |

## Dependencies

- **Python 3.8+**
- **numpy** — matrix operations
- **pytest** — testing
