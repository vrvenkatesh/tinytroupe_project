import random
from dataclasses import dataclass
from typing import Dict, Any, List

from models.enums import Region
from agents.base import Agent

@dataclass
class ExternalEventAgent(Agent):
    """External event generator agent."""
    event_type: str  # 'weather', 'geopolitical', 'market'

    def generate_event(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an external event based on type and current state."""
        event = {
            'type': self.event_type,
            'severity': self._calculate_severity(world_state),
            'duration': self._calculate_duration(world_state),
            'affected_regions': self._determine_affected_regions(world_state),
        }
        return event

    def _calculate_severity(self, world_state: Dict[str, Any]) -> float:
        """Calculate event severity based on type and conditions."""
        base_severity = self.config['severity']
        type_factor = {
            'weather': 0.7,
            'geopolitical': 0.9,
            'market': 0.5,
        }[self.event_type]
        return base_severity * type_factor

    def _calculate_duration(self, world_state: Dict[str, Any]) -> int:
        """Calculate event duration in days."""
        base_duration = {
            'weather': 7,
            'geopolitical': 30,
            'market': 90,
        }[self.event_type]
        return int(base_duration * (0.5 + random.random() * 0.5))

    def _determine_affected_regions(self, world_state: Dict[str, Any]) -> List[Region]:
        """Determine which regions are affected by the event."""
        if self.event_type == 'weather':
            return [random.choice(list(Region))]
        elif self.event_type == 'geopolitical':
            return [r for r in Region if random.random() < 0.3]
        else:  # market
            return list(Region) 