import min_pair_sat_scalability
from min_pair_sat_scalability.cli import main


def test_version():
    assert min_pair_sat_scalability.__version__


def test_cli_runs():
    assert main([]) == 0
