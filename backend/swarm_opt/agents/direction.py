from __future__ import annotations

from ..features.targets import DIRECTION_CLASSES
from .base import XGBClassifierAgent


class DirectionAgent(XGBClassifierAgent):
    AGENT_NAME = "direction"
    CLASSES = DIRECTION_CLASSES
    TARGET_DEFINITION = (
        "3-class forward weekly return close[exit]/close[decision]-1 vs a "
        "threshold fit as 0.5*std(TRAIN forward returns); see "
        "swarm_opt.features.targets.direction_label"
    )
