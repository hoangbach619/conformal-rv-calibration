"""End-to-end integration test, gated behind --slow.

This is a placeholder for the full walk-forward pipeline: load data, build
features, fit the base model, construct CQR intervals, apply each online
correction, and compute the regime-conditional coverage gap. It is marked
slow so it never runs in the fast CI subset, and it is skipped (not failed)
until the pipeline is implemented.
"""

from __future__ import annotations

import pytest


@pytest.mark.slow
def test_full_walk_forward_pipeline() -> None:
    pytest.skip("pipeline not implemented; design frozen in preregistration")
