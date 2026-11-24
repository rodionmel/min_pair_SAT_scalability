import itertools
import os

import pytest

from min_pair_sat_scalability.cli import main
from min_pair_sat_scalability.generators import gnp, power, triangular, triangular_lattice
from min_pair_sat_scalability.graph import Graph, read_graph


def edge_set(g):
    return set(g.edges)


# --- G(n, p) ----------------------------------------------------------------

def test_gnp_extremes():
    assert gnp(10, 0.0).m == 0
    assert gnp(10, 1.0).m == 45


def test_gnp_deterministic_per_seed():
    assert gnp(30, 0.3, seed=7) == gnp(30, 0.3, seed=7)
    assert gnp(30, 0.3, seed=7) != gnp(30, 0.3, seed=8)


def test_gnp_density_close_to_p():
    g = gnp(200, 0.3, seed=1)
    assert abs(g.m / (200 * 199 / 2) - 0.3) < 0.02


def test_bad_probability_rejected():
    with pytest.raises(ValueError):
        gnp(5, 1.5)


# --- triangular lattice ------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3, 6])
def test_lattice_size(n):
    g = triangular_lattice(n)
    assert g.n == n * (n + 1) // 2
    assert g.m == 3 * n * (n - 1) // 2
    assert max((len(a) for a in g.adjacency()), default=0) <= 6


def test_lattice_n3_exact():
    #        0
    #       1 2
    #      3 4 5
    assert edge_set(triangular_lattice(3)) == {(0, 1), (0, 2), (1, 2), (1, 3), (1, 4),
                                               (2, 4), (2, 5), (3, 4), (4, 5)}


def test_power_of_path():
    path = Graph.from_edges(5, [(i, i + 1) for i in range(4)])
    assert power(path, 1) == path
    assert edge_set(power(path, 2)) == edge_set(path) | {(0, 2), (1, 3), (2, 4)}
    assert power(path, 4).m == 10  # diameter reached: complete graph


def test_power_does_not_bridge_components():
    g = Graph.from_edges(4, [(0, 1), (2, 3)])
    assert power(g, 5) == g


def test_triangular_no_exclusion_is_lattice():
    assert triangular(5, 0.0) == triangular_lattice(5)


def test_triangular_full_exclusion_is_empty():
    g = triangular(5, 1.0, e=3)
    assert (g.n, g.m) == (15, 0)


def test_triangular_power_is_taken_after_exclusion():
    base = triangular(6, 0.3, e=1, seed=4)
    assert triangular(6, 0.3, e=2, seed=4) == power(base, 2)  # same seed -> same exclusion


def test_triangular_large_power_gives_complete_graph():
    n = 4  # diameter of the full lattice is n-1
    g = triangular(n, 0.0, e=n - 1)
    assert g.m == g.n * (g.n - 1) // 2


def test_triangular_exclusion_rate():
    g = triangular(40, 0.25, seed=2)
    assert abs(g.m / triangular_lattice(40).m - 0.75) < 0.03


# --- CLI --------------------------------------------------------------------

def test_cli_gen_writes_product_of_params(tmp_path):
    out = tmp_path / "g"
    assert main(["gen", "triangular", "--n", "4", "5", "--p", "0.1", "--e", "1", "2",
                 "--count", "3", "--out-dir", str(out)]) == 0
    files = sorted(os.listdir(out))
    assert len(files) == 2 * 1 * 2 * 3
    assert "tri_n5_p0.1_e2_s2.dimacs" in files
    g = read_graph(out / "tri_n5_p0.1_e2_s2.dimacs")
    assert g == triangular(5, 0.1, e=2, seed=2)


def test_cli_gen_gnp(tmp_path):
    assert main(["gen", "gnp", "--n", "12", "--p", "0.5", "--seed", "4", "--out-dir", str(tmp_path)]) == 0
    assert read_graph(tmp_path / "gnp_n12_p0.5_s4.dimacs") == gnp(12, 0.5, seed=4)
