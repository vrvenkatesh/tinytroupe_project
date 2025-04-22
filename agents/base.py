from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class BaseAgent:
    """Base class for all agents in the simulation."""
    name: str
    config: Dict[str, Any]
    simulation_id: str 