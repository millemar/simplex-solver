"""
Netlib LP test problem downloader.

Downloads MPS files from https://netlib.org/lp/data/ and caches them
locally in the `data/` directory.
"""

import gzip
import os
import urllib.request
from typing import Optional

# Base URL for Netlib LP data
NETLIB_BASE_URL = "https://netlib.org/lp/data/"

# Local cache directory
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Known Netlib problem names for reference
NETLIB_PROBLEMS = [
    "afiro", "adlittle", "blend", "kb2", "sc50a", "sc50b",
    "sc105", "share2b", "share1b", "stocfor1", "lotfi",
    "boeing1", "boeing2", "e226", "etamacro", "fffff800",
]


def get_problem_path(name: str) -> str:
    """
    Return the local path for a cached problem, downloading it if needed.

    Parameters
    ----------
    name : str
        Netlib problem name (e.g. 'afiro') or path to an existing MPS file.

    Returns
    -------
    str
        Path to the local MPS file (plain text, not compressed).
    """
    # If it's already a file path that exists, return it directly
    if os.path.isfile(name):
        return name

    # Otherwise treat as a Netlib problem name
    os.makedirs(DATA_DIR, exist_ok=True)

    local_mps = os.path.join(DATA_DIR, f"{name}.mps")
    if os.path.isfile(local_mps):
        return local_mps

    # Try to download
    return _download_problem(name)


def _download_problem(name: str) -> str:
    """
    Download a Netlib problem and return the local MPS file path.

    Tries both plain and gzipped formats.

    Parameters
    ----------
    name : str
        Netlib problem name (lowercase).

    Returns
    -------
    str
        Path to the downloaded and decompressed MPS file.

    Raises
    ------
    RuntimeError
        If the problem cannot be downloaded.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    local_mps = os.path.join(DATA_DIR, f"{name}.mps")

    # Try gzipped first
    gz_url = f"{NETLIB_BASE_URL}{name}.gz"
    plain_url = f"{NETLIB_BASE_URL}{name}"

    for url, is_gz in [(gz_url, True), (plain_url, False)]:
        try:
            print(f"Downloading {url} ...")
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "simplex-solver/1.0"},
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                raw = response.read()

            if is_gz:
                content = gzip.decompress(raw).decode("utf-8", errors="replace")
            else:
                content = raw.decode("utf-8", errors="replace")

            with open(local_mps, "w") as f:
                f.write(content)

            print(f"  Saved to {local_mps}")
            return local_mps

        except Exception as e:
            print(f"  Failed ({e}), trying next URL...")
            continue

    raise RuntimeError(
        f"Could not download Netlib problem '{name}'. "
        f"Tried: {gz_url} and {plain_url}"
    )


def download_all(problems: Optional[list] = None) -> None:
    """
    Download a list of Netlib problems (or all standard ones).

    Parameters
    ----------
    problems : list of str, optional
        List of problem names to download. Defaults to NETLIB_PROBLEMS.
    """
    if problems is None:
        problems = NETLIB_PROBLEMS

    for name in problems:
        try:
            path = get_problem_path(name)
            print(f"  {name}: {path}")
        except Exception as e:
            print(f"  {name}: FAILED — {e}")
