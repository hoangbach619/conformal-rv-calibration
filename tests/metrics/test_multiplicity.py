"""Unit tests for the Benjamini-Hochberg multiplicity control.

A canonical worked example with the correct subset rejected at FDR 0.10 and
monotone adjusted p-values, including in a shuffled order. None touch the
network.
"""

from __future__ import annotations

import numpy as np
import pytest

from conformal_rv.metrics import multiplicity


def test_fdr_level_is_pre_registered() -> None:
    assert multiplicity.FDR_LEVEL == 0.10


def test_benjamini_hochberg_worked_example() -> None:
    # Critical values i/5 * 0.10 = [0.02, 0.04, 0.06, 0.08, 0.10]; the largest
    # ranked p below its line is 0.04 (rank 4), so the first four are rejected.
    pvalues = np.array([0.01, 0.02, 0.03, 0.04, 0.20])
    result = multiplicity.benjamini_hochberg(pvalues, fdr=0.10)

    assert result.rejected.tolist() == [True, True, True, True, False]
    # Adjusted (q-) values: 5/j * p_(j) is 0.05 for the first four, 0.20 for the
    # last, then the running minimum leaves them unchanged.
    assert np.allclose(result.adjusted_pvalues, [0.05, 0.05, 0.05, 0.05, 0.20])


def test_benjamini_hochberg_respects_input_order() -> None:
    # The same p-values shuffled: rejections and adjusted values track position.
    pvalues = np.array([0.04, 0.20, 0.01, 0.03, 0.02])
    result = multiplicity.benjamini_hochberg(pvalues, fdr=0.10)

    assert result.rejected.tolist() == [True, False, True, True, True]
    assert np.allclose(result.adjusted_pvalues, [0.05, 0.20, 0.05, 0.05, 0.05])
    # Adjusted values are monotone non-decreasing once ordered by p-value.
    order = np.argsort(pvalues)
    assert np.all(np.diff(result.adjusted_pvalues[order]) >= 0.0)


def test_effective_number_of_tests_is_a_later_phase() -> None:
    with pytest.raises(NotImplementedError):
        multiplicity.effective_number_of_tests(np.array([0.1, 0.2]))
