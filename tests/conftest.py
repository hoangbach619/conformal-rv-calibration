"""Pytest configuration: the --slow gate.

Slow tests (notably the end-to-end integration test) are skipped by default so
the fast subset stays fast in CI. They run only when --slow is passed. This
keeps the default ``pytest`` invocation equal to the fast subset.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="run slow tests, including the end-to-end integration test",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: marks a test as slow")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--slow"):
        return
    skip_slow = pytest.mark.skip(reason="need --slow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
