import itertools
import random

import pytest

from min_pair_sat_scalability.graph import Graph
from min_pair_sat_scalability.problems import check_pair_cover, encode
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


def solve(g, **kw):
    return minimize(lambda k: encode("pair_cover", g, k), **kw)


# Optimal binary covering arrays: K_2, K_3 -> 4, K_4 -> 5, K_5..K_10 -> 6
# (Kleitman-Spencer); bipartite graphs, odd cycles and the Petersen graph -> 4.
# K_11 (-> 7) is left out: the basic encoding cannot prove K=6 UNSAT quickly.
KNOWN = [(complete(2), 4), (complete(3), 4), (complete(4), 5), (complete(5), 6),
         (complete(10), 6), (cycle(5), 4), (cycle(6), 4), (petersen(), 4)]


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("graph,expected", KNOWN, ids=[g.name for g, _ in KNOWN])
def test_known_optima(graph, expected, strategy):
    res = solve(graph, strategy=strategy)
    assert (res.status, res.cost, res.lower_bound) == ("optimal", expected, expected)
    assert len(res.solution["rows"]) == expected
    assert check_pair_cover(graph, res.solution["rows"]) == []


@pytest.mark.parametrize("seed", range(6))
def test_strategies_agree_on_random_graphs(seed):
    g = random_graph(12, 0.5, seed)
    lin, bis = solve(g, strategy="linear"), solve(g, strategy="binary")
    assert lin.status == bis.status == "optimal"
    assert lin.cost == bis.cost
    assert check_pair_cover(g, bis.solution["rows"]) == []


def test_linear_proof_is_all_smaller_k_unsat():
    res = solve(complete(4), strategy="linear")
    assert [(c["k"], c["result"]) for c in res.calls] == [(0, False), (1, False), (2, False),
                                                         (3, False), (4, False), (5, True)]


def test_binary_doubles_then_bisects():
    res = solve(complete(4), strategy="binary")
    assert [(c["k"], c["result"]) for c in res.calls] == [(0, False), (1, False), (2, False),
                                                         (4, False), (8, True), (6, True), (5, True)]


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("k_max", [0, 3, 4, 6])
def test_k_max_cap(strategy, k_max):
    res = solve(complete(4), strategy=strategy, k_max=k_max)  # optimum is 5
    if k_max < 5:
        assert (res.status, res.cost, res.lower_bound) == ("unsat", None, k_max + 1)
    else:
        assert (res.status, res.cost) == ("optimal", 5)


def test_edgeless_graph_needs_no_stages():
    res = solve(Graph(3, ()))
    assert (res.status, res.cost, res.solution["rows"]) == ("optimal", 0, [])


def test_self_loop_rejected():
    with pytest.raises(ValueError):
        encode("pair_cover", Graph(2, ((0, 0),)), 4)


def test_encoding_size():
    # x: n*k vars; per edge 4 patterns * k witnesses, each with 2 binary clauses, + 1 clause per pattern
    g, k = complete(3), 5
    enc = encode("pair_cover", g, k)
    assert enc.nvars == g.n * k + g.m * 4 * k
    assert len(enc.clauses) == g.m * 4 * (2 * k + 1)


def test_checker_reports_missing():
    assert check_pair_cover(complete(2), ["00", "01", "10"]) == [((0, 1), "11")]


def test_timeout():
    res = solve(complete(20), timeout=0.0)
    assert res.status == "timeout"


def test_non_interruptible_solver_with_timeout_rejected():
    with pytest.raises(ValueError, match="cannot be interrupted"):
        solve(complete(3), solver="cadical195", timeout=10)


def test_cadical_without_timeout_works():
    assert solve(complete(4), solver="cadical195").cost == 5


# --- optional symmetry breaking -------------------------------------------------

def test_lex_leq_matches_brute_force():
    from pysat.formula import IDPool
    from pysat.solvers import Solver
    from min_pair_sat_scalability.problems import lex_leq
    L = 3
    for strict in (False, True):
        for a_bits in itertools.product((0, 1), repeat=L):
            for b_bits in itertools.product((0, 1), repeat=L):
                pool = IDPool()
                a = [pool.id(("a", i)) for i in range(L)]
                b = [pool.id(("b", i)) for i in range(L)]
                units = [[v if bit else -v] for v, bit in zip(a + b, a_bits + b_bits)]
                with Solver("glucose4", bootstrap_with=lex_leq(pool, a, b, strict) + units) as s:
                    expected = a_bits < b_bits if strict else a_bits <= b_bits
                    assert s.solve() == expected, (a_bits, b_bits, strict)


@pytest.mark.parametrize("symmetry", ["zero", "zero_lex"])
@pytest.mark.parametrize("graph,expected", KNOWN, ids=[g.name for g, _ in KNOWN])
def test_symmetry_modes_keep_optimum(graph, expected, symmetry):
    res = minimize(lambda k: encode("pair_cover", graph, k, symmetry=symmetry))
    assert (res.status, res.cost) == ("optimal", expected)
    rows = res.solution["rows"]
    assert check_pair_cover(graph, rows) == []
    assert set(rows[0]) <= {"0"}  # stage 0 all zeros
    if symmetry == "zero_lex":
        assert rows[1:] == sorted(rows[1:])


@pytest.mark.parametrize("seed", range(6))
def test_symmetry_modes_agree_on_random_graphs(seed):
    g = random_graph(12, 0.5, seed)
    costs = {s: minimize(lambda k: encode("pair_cover", g, k, symmetry=s)).cost
             for s in ("none", "zero", "zero_lex")}
    assert len(set(costs.values())) == 1, costs


def test_col_lex_on_complete_graphs():
    for n, expected in [(3, 4), (4, 5), (6, 6), (10, 6)]:
        g = complete(n)
        res = minimize(lambda k: encode("pair_cover", g, k, symmetry="zero_lex", col_lex=True))
        assert res.cost == expected
        cols = ["".join(r[v] for r in res.solution["rows"][1:]) for v in range(n)]
        assert cols == sorted(cols) and len(set(cols)) == n


def test_unknown_symmetry_rejected():
    with pytest.raises(ValueError):
        encode("pair_cover", complete(2), 4, symmetry="bogus")
