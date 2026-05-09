# Copyright (c) 2026 CortexFlow / Adrian Creox. All rights reserved.
# Licensed under the Apache License, Version 2.0

"""
CortexFlow — The Cognitive Operating System for AI Agents.
"""

__version__ = "0.1.0"
__author__ = "Adrian Creox"
__license__ = "Apache-2.0"
__copyright__ = "Copyright (c) 2026 CortexFlow Authors"

from cortexflow.sdk.agent import defineAgent
from cortexflow.sdk.swarm import defineTeam
from cortexflow.core.runtime import CortexRuntime
from cortexflow.core.config import get_config

__all__ = ["defineAgent", "defineTeam", "CortexRuntime", "get_config"]
