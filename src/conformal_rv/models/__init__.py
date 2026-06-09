"""Point and quantile forecasters for realised volatility.

HAR-RV and quantile-regression HAR are the credible base. The AR/linear model
is a sanity baseline. The single deep comparator is a calibration stress test
only, not a contender for best forecaster.
"""

from __future__ import annotations
