import itertools
import random

import pytest

from min_pair_sat_scalability.graph import Graph
from min_pair_sat_scalability.problems import (can2, check_pair_cover, encode, greedy_clique,
                                               greedy_coloring)
from min_pair_sat_scalability.solve import STRATEGIES, minimize


def complete(n):
    return Graph.from_edges(n, itertools.combinations(range(n), 2), name=f"K{n}")


def cycle(n):
    return Graph.from_edges(n, [(i, (i + 1) % n) for i in range(n)], name=f"C{n}")


def petersen():
    outer = [(i, (i + 1) % 5) for i in range(5)]
    inner = [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
    spokes = [(i, i + 5) for i in range(5)]
    return Graph.from_edges(10, outer + inner + spokes, name="petersen")


def random_graph(n, p, seed):
    rng = random.Random(seed)
    return Graph.from_edges(n, [e for e in itertools.combinations(range(n), 2) if rng.random() < p])


def test_can2_known_values():
    # Kleitman-Spencer: N=4 covers 3 columns, 5 -> 4, 6 -> 10, 7 -> 15, 8 -> 35
    assert [can2(k) for k in (2, 3, 4, 5, 10, 11, 15, 16, 35, 36)] == [4, 4, 5, 6, 6, 7, 7, 8, 8, 9]


def test_greedy_helpers_are_valid():
    g = random_graph(30, 0.3, 0)
    col = greedy_coloring(g)
    assert all(col[i] != col[j] for i, j in g.edges)
    adj = [set(a) for a in g.adjacency()]
    cl = greedy_clique(g)
    assert all(b in adj[a] for a, b in itertools.combinations(cl, 2))


KNOWN = [(complete(2), 4), (complete(3), 4), (complete(4), 5), (complete(5), 6),
         (complete(10), 6), (complete(11), 7), (cycle(5), 4), (cycle(6), 4), (petersen(), 4)]


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("graph,expected", KNOWN, ids=[g.name for g, _ in KNOWN])
def test_known_optima(graph, expected, strategy):
    res = minimize(encode("pair_cover", graph), strategy=strategy)
    assert (res.status, res.cost) == ("optimal", expected)
    assert res.solution["stages"] == expected
    assert check_pair_cover(graph, res.solution["rows"]) == []


@pytest.mark.parametrize("seed", range(6))
def test_variants_agree_on_random_graphs(seed):
    g = random_graph(14, 0.5, seed)
    base = minimize(encode("pair_cover", g))
    # no symmetry breaking, no clique bound, oversized stage budget: search must still find it
    plain = minimize(encode("pair_cover", g, zero_stage=False, bounds=False, k_max=8), strategy="binary")
    assert base.status == plain.status == "optimal"
    assert base.cost == plain.cost
    for r in (base, plain):
        assert check_pair_cover(g, r.solution["rows"]) == []


def test_bounds_bracket_optimum():
    g = random_graph(25, 0.4, 3)
    enc = encode("pair_cover", g)
    res = minimize(enc)
    assert enc.info["lb"] <= res.cost <= enc.info["ub"]


def test_too_few_stages_is_unsat():
    assert minimize(encode("pair_cover", complete(4), k_max=4)).status == "unsat"


def test_edgeless_graph_needs_no_stages():
    res = minimize(encode("pair_cover", Graph(3, ())))
    assert (res.status, res.cost) == ("optimal", 0)


def test_self_loop_rejected():
    with pytest.raises(ValueError):
        encode("pair_cover", Graph(2, ((0, 0),)))


def test_checker_reports_missing():
    g = complete(2)
    assert check_pair_cover(g, ["00", "01", "10"]) == [((0, 1), "11")]
