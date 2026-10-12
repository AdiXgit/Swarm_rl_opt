from __future__ import annotations

from ..features.targets import VOLATILITY_CLASSES
from .base import XGBClassifierAgent


class VolatilityAgent(XGBClassifierAgent):
    AGENT_NAME = "volatility"
    CLASSES = VOLATILITY_CLASSES
    TARGET_DEFINITION = (
        "3-class forward/baseline realized-volatility ratio over the episode's "
        "mark window vs fixed 1.10/0.90 bands; see "
        "swarm_opt.features.targets.volatility_label"
    )
