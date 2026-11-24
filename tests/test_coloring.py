import itertools
import json

import pytest

from min_pair_sat_scalability import universal
from min_pair_sat_scalability.cli import main
from min_pair_sat_scalability.graph import Graph
from min_pair_sat_scalability.problems import check_pair_cover, encode, greedy_clique, k_min
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


def qi5():
    """2- and 3-subsets of {1..4}, adjacent iff qualitatively independent:
    chi = 5 but pair cover needs only 5 stages (colouring route gives 6)."""
    sets = [frozenset(c) for r in (2, 3) for c in itertools.combinations(range(1, 5), r)]
    edges = [(i, j) for i, j in itertools.combinations(range(10), 2)
             if sets[i] - sets[j] and sets[j] - sets[i] and sets[i] & sets[j]]
    return Graph.from_edges(10, edges, name="qi5")


def color(g, **kw):
    return minimize(lambda k: encode("coloring", g, k), k_min=k_min("coloring", g), **kw)


def test_greedy_clique_is_a_clique():
    for g in (complete(6), petersen(), qi5(), cycle(7)):
        adj = [set(a) for a in g.adjacency()]
        cl = greedy_clique(g)
        assert all(b in adj[a] for a, b in itertools.combinations(cl, 2))
    assert len(greedy_clique(complete(6))) == 6


# (graph, chi, stages from the universal set of size chi)
CASES = [(complete(2), 2, 4), (complete(4), 4, 5), (complete(10), 10, 6), (cycle(5), 3, 4),
         (cycle(6), 2, 4), (petersen(), 3, 4), (qi5(), 5, 6)]


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("graph,chi,stages", CASES, ids=[c[0].name for c in CASES])
def test_coloring_route(graph, chi, stages, strategy):
    res = color(graph, strategy=strategy)
    sol = res.solution
    assert (res.status, res.cost, sol["chi"], sol["stages"]) == ("optimal", chi, chi, stages)
    assert all(sol["colors"][u] != sol["colors"][v] for u, v in graph.edges)
    assert [sol["colors"][v] for v in sol["clique"]] == list(range(len(sol["clique"])))  # clique fixing
    assert check_pair_cover(graph, sol["rows"]) == []


def test_coloring_is_only_an_upper_bound():
    g = qi5()
    exact = minimize(lambda k: encode("pair_cover", g, k))
    assert exact.cost == 5 < color(g).solution["stages"] == 6


def test_search_starts_at_clique_size():
    res = color(complete(5))
    assert [c["k"] for c in res.calls] == [5]  # the clique proves k < 5 infeasible


def test_edgeless_and_empty_graphs():
    sol = color(Graph(3, ())).solution
    assert (sol["chi"], sol["stages"], sol["rows"]) == (1, 0, [])
    assert color(Graph(0, ())).cost == 0


def test_missing_universal_set_reports_colouring_only(tmp_path):
    g = complete(3)
    res = minimize(lambda k: encode("coloring", g, k, universal_sets=str(tmp_path / "none.json")),
                   k_min=3)
    assert res.solution["chi"] == 3 and res.solution["stages"] is None


def test_packaged_table_matches_known_optima():
    # Kleitman-Spencer: least K with C(K-1, ceil(K/2)) >= c
    table = universal.load_table()
    assert {c: table[c]["k"] for c in range(1, 11)} == {1: 0, 2: 4, 3: 4, 4: 5, 5: 6, 6: 6, 7: 6,
                                                       8: 6, 9: 6, 10: 6}
    for c in range(2, 11):
        cols = table[c]["columns"]
        rows = ["".join(col[s] for col in cols) for s in range(table[c]["k"])]
        assert check_pair_cover(complete(c), rows) == []


def test_precompute_resumes_and_skips(tmp_path):
    path = tmp_path / "u.json"
    universal.precompute(1, 3, path=path)
    assert sorted(universal.load_table(path)) == [1, 2, 3]
    universal.precompute(2, 5, path=path)  # 2, 3 skipped
    assert {c: e["k"] for c, e in universal.load_table(path).items()} == {1: 0, 2: 4, 3: 4, 4: 5, 5: 6}


def test_cli_universal_and_coloring_job(tmp_path, capsys):
    path = tmp_path / "u.json"
    assert main(["universal", "--c", "3", "--table", str(path)]) == 0
    assert main(["universal", "--show", "--table", str(path)]) == 0
    assert "c=  3  K=4" in capsys.readouterr().out
    gfile = tmp_path / "c5.txt"
    gfile.write_text("0 1\n1 2\n2 3\n3 4\n4 0\n")
    assert main(["solve", str(gfile), "--problem", "coloring"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert (out["cost"], out["solution"]["stages"]) == (3, 4)


def test_fill_stages_redecodes_missing(tmp_path):
    from min_pair_sat_scalability import jobs
    from min_pair_sat_scalability.graph import write_graph
    write_graph(petersen(), tmp_path / "p.dimacs")
    spec = tmp_path / "s.json"
    spec.write_text(json.dumps({"problem": "coloring", "graphs": ["p.dimacs"]}))
    run = tmp_path / "run"
    jobs.plan(spec, run)
    jobs.run_local(run)
    path = jobs.result_path(run, 0)
    r = json.loads(path.read_text())
    r["solution"]["stages"] = r["solution"]["rows"] = None  # as if chi was beyond the table
    path.write_text(json.dumps(r))
    assert jobs.fill_stages(run) == 1
    sol = json.loads(path.read_text())["solution"]
    assert sol["stages"] == 4 and check_pair_cover(petersen(), sol["rows"]) == []
