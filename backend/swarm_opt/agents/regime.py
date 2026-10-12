from __future__ import annotations

from ..features.targets import REGIME_CLASSES
from .base import XGBClassifierAgent


class RegimeAgent(XGBClassifierAgent):
    AGENT_NAME = "regime"
    CLASSES = REGIME_CLASSES
    TARGET_DEFINITION = (
        "5-class mutually-exclusive priority rule over forward return and "
        "forward/baseline volatility ratio (vol extremes checked before "
        "trend); see swarm_opt.features.targets.regime_label"
    )
