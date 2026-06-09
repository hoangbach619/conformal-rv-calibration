"""Tests for the data module stubs."""

from __future__ import annotations

import pytest

from conformal_rv import data


def test_loaders_are_present() -> None:
    assert callable(data.load_index_ohlc)
    assert callable(data.load_oxford_man_rv)
    assert callable(data.assert_no_backfill)


def test_load_index_ohlc_not_yet_implemented() -> None:
    with pytest.raises(NotImplementedError):
        data.load_index_ohlc("^GSPC", "2000-01-01", "2020-01-01")
