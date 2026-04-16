"""
Tests for the two-phase simplex solver against known Netlib optimal values.

Known optimal objective values (from Netlib):
    afiro    -4.6475314286e+02
    adlittle  2.2549496316e+05
    sc50a    -6.4575077059e+01
    sc50b    -7.0000000000e+01
    kb2      -1.7499001299e+03
"""

import math
import os
import socket
import sys
import pytest
import numpy as np

# Make sure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mps_parser import parse_mps, LPProblem
from simplex import solve, solve_linear_program
from netlib_downloader import get_problem_path

# Relative tolerance for comparing objective values
REL_TOL = 1e-4

# Known optimal values
KNOWN_OPTIMA = {
    "afiro":    -4.6475314286e+02,
    "adlittle":  2.2549496316e+05,
    "sc50a":   -6.4575077059e+01,
    "sc50b":   -7.0000000000e+01,
    "kb2":     -1.7499001299e+03,
}

# Directory containing bundled test MPS files
TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _is_network_available() -> bool:
    """Check whether external network (netlib.org) is reachable."""
    try:
        socket.setdefaulttimeout(5)
        socket.getaddrinfo("netlib.org", 80)
        return True
    except OSError:
        return False


requires_network = pytest.mark.skipif(
    not _is_network_available(),
    reason="netlib.org not reachable (no network)",
)


# ===================================================================== #
#  Local (offline) unit tests                                             #
# ===================================================================== #

class TestSmallLP:
    """Tests on tiny LPs that don't require network access."""

    def test_simple_mps(self):
        """Parse and solve a simple two-constraint MPS file."""
        path = os.path.join(TEST_DATA_DIR, "simple.mps")
        lp = parse_mps(path)
        result = solve(lp)
        assert result.status == "optimal"
        # min -x1-x2, x1+x2<=2, x1-x2<=1 -> optimal at (1.5,0.5), obj=-2
        assert math.isclose(result.objective, -2.0, rel_tol=REL_TOL)

    def test_solve_linear_program_from_file(self):
        """Solve using the public function with a local MPS data file."""
        path = os.path.join(TEST_DATA_DIR, "simple.mps")
        result = solve_linear_program(data_file=path)
        assert result.status == "optimal"
        assert math.isclose(result.objective, -2.0, rel_tol=REL_TOL)

    def test_equality_mps(self):
        """Parse and solve an MPS file with an equality constraint."""
        path = os.path.join(TEST_DATA_DIR, "equality.mps")
        lp = parse_mps(path)
        result = solve(lp)
        assert result.status == "optimal"
        # min -2x1-x2, x1+x2=3, x1+2x2<=4 -> x1=3,x2=0, obj=-6
        assert math.isclose(result.objective, -6.0, rel_tol=REL_TOL)

    def test_tiny_lp_direct(self):
        """Build a tiny LP directly and solve it (no MPS parsing)."""
        # min -x1-2x2 s.t. x1+x2<=4, x1-x2<=2, x1,x2>=0
        # Standard form: x1+x2+s1=4, x1-x2+s2=2
        # Optimal at (x1=0, x2=4): obj=-8 (feasible: x1-x2=-4<=2 ✓, x1+x2=4 ✓)
        A = np.array([[1, 1, 1, 0], [1, -1, 0, 1]], dtype=float)
        b = np.array([4.0, 2.0])
        c = np.array([-1.0, -2.0, 0.0, 0.0])
        lp = LPProblem("tiny", c, A, b, ["x1","x2","s1","s2"], ["c1","c2"], n_orig=2)
        result = solve(lp)
        assert result.status == "optimal"
        assert math.isclose(result.objective, -8.0, rel_tol=REL_TOL)

    def test_two_constraint_lp(self):
        """min -x1-x2 with two inequality constraints."""
        A = np.array([[1, 1, 1, 0], [1, 0, 0, 1]], dtype=float)
        b = np.array([2.0, 1.0])
        c = np.array([-1.0, -1.0, 0.0, 0.0])
        lp = LPProblem("test2c", c, A, b, ["x1","x2","s1","s2"], ["c1","c2"], n_orig=2)
        result = solve(lp)
        assert result.status == "optimal"
        assert math.isclose(result.objective, -2.0, rel_tol=REL_TOL)

    def test_infeasible_lp(self):
        """A clearly infeasible LP: x<=1 and x>=3 simultaneously."""
        # x <= 1: x + s1 = 1 (b1=1 >= 0)
        # x >= 3: x - s2 = 3, but with b[1]=-3 < 0, flip row -> -x + s2 = 3
        A = np.array([[1, 1, 0], [-1, 0, 1]], dtype=float)
        b = np.array([1.0, -3.0])
        A[1, :] *= -1
        b[1] *= -1
        # Now A = [[1,1,0],[1,0,-1]], b=[1,3]
        c = np.array([1.0, 0.0, 0.0])
        lp = LPProblem("infeas", c, A, b, ["x","s1","s2"], ["c1","c2"], n_orig=1)
        result = solve(lp)
        assert result.status == "infeasible"

    def test_unbounded_lp(self):
        """An unbounded LP: minimize -x with x >= 0 (no upper bound)."""
        # min -x, s.t. x + s = 5 (from x<=5)... actually:
        # For unbounded, we need no upper bound on the objective direction.
        # min -x, x <= 5 is bounded at -5. Let's do:
        # min -x1-x2, x1-x2<=1, no upper bound on x1+x2
        # This is unbounded: go to x1=inf, x2=inf along x1-x2=1
        A = np.array([[1, -1, 1]], dtype=float)
        b = np.array([1.0])
        c = np.array([-1.0, -1.0, 0.0])
        lp = LPProblem("unbound", c, A, b, ["x1","x2","s1"], ["c1"], n_orig=2)
        result = solve(lp)
        assert result.status == "unbounded"

    def test_klee_minty_3d_direct(self):
        """
        Klee-Minty 3D cube in standard form (with slacks).
        Max x1 + 100x2 + 10000x3  <=>  min -(x1 + 100x2 + 10000x3)
        """
        A = np.array(
            [
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [4.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                [8.0, 4.0, 1.0, 0.0, 0.0, 1.0],
            ]
        )
        b = np.array([5.0, 25.0, 125.0])
        c = np.array([-1.0, -100.0, -10000.0, 0.0, 0.0, 0.0])

        result = solve_linear_program(A=A, b=b, c=c)
        assert result.status == "optimal"
        assert math.isclose(result.objective, -1.25e6, rel_tol=REL_TOL)


# ===================================================================== #
#  Netlib tests (require network access)                                  #
# ===================================================================== #

def _solve_netlib(name: str) -> float:
    """Download (if needed), parse, and solve a Netlib problem. Returns objective."""
    path = get_problem_path(name)
    result = solve_linear_program(data_file=path, verbose=False)
    assert result.status == 'optimal', f"{name}: expected optimal, got {result.status}"
    return result.objective


@requires_network
@pytest.mark.parametrize("name,expected", list(KNOWN_OPTIMA.items()))
def test_netlib_optimal(name, expected):
    """Solve each Netlib problem and check against the known optimal value."""
    obj = _solve_netlib(name)
    if expected != 0.0:
        rel_err = abs(obj - expected) / abs(expected)
    else:
        rel_err = abs(obj - expected)
    assert rel_err <= REL_TOL, (
        f"{name}: objective={obj:.10e}, expected={expected:.10e}, "
        f"rel_err={rel_err:.2e} > {REL_TOL}"
    )


@requires_network
def test_afiro():
    """Dedicated test for afiro (most commonly cited benchmark)."""
    obj = _solve_netlib("afiro")
    assert math.isclose(obj, -4.6475314286e+02, rel_tol=REL_TOL), (
        f"afiro: got {obj:.10e}, expected -4.6475314286e+02"
    )


@requires_network
def test_sc50b():
    """Dedicated test for sc50b."""
    obj = _solve_netlib("sc50b")
    assert math.isclose(obj, -7.0000000000e+01, rel_tol=REL_TOL), (
        f"sc50b: got {obj:.10e}, expected -7.0000000000e+01"
    )
