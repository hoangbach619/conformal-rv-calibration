"""Experiment engine: wiring the pieces into single-configuration runs.

A configuration is one index, one horizon and one seed. The engine drives the
existing data, realised-vol, feature, split, model and conformal modules through
a walk-forward and pools the out-of-sample test folds into one result per
conformal method, with aligned regime labels and days-since-break.
"""

from __future__ import annotations
