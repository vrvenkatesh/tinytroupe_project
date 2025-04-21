"""External event generator agent for supply chain simulation."""

from typing import Dict, Any, Union, List
from dataclasses import dataclass
import random
import logging

from tinytroupe.agent import TinyPerson
from models.enums import Region
from tinytroupe.utils import post_init

logger = logging.getLogger(__name__)

@dataclass
class ExternalEventAgent(TinyPerson):
    """Agent that generates external events in the simulation."""
    event_type: str  # 'weather', 'geopolitical', 'market'
    config: Dict[str, Any]

    def __init__(self, name: str, config: Dict[str, Any], simulation_id: str, event_type: str):
        """Initialize the external event agent."""
        super().__init__(name=name)
        
        self.event_type = event_type
        self.config = config
        self.simulation_id = simulation_id
        self.seed = hash(self.simulation_id + self.event_type) % (2**32)
        random.seed(self.seed)

        # Validate event type
        if event_type not in ['weather', 'geopolitical', 'market']:
            raise ValueError(f"Invalid event type: {event_type}")

    def _post_init(self, **kwargs):
        """Post-initialization setup."""
        super()._post_init(**kwargs)
        if hasattr(self, 'simulation_id'):
            self._simulation_id = self.simulation_id
            delattr(self, 'simulation_id')

    def _calculate_severity(self, world_state: Dict[str, Any]) -> float:
        """Calculate event severity based on type and conditions."""
        base_severity = self.config.get('severity', 1.0)
        type_factor = {
            'weather': 0.7,
            'geopolitical': 0.9,
            'market': 0.5,
        }[self.event_type]
        return base_severity * type_factor * random.uniform(0.8, 1.2)

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

    def generate_event(self, world_state: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate a random external event based on configuration and world state."""
        if world_state is None:
            world_state = {}

        if random.random() < self.config.get('frequency', 0.1):
            event = {
                'type': self.event_type,
                'severity': self._calculate_severity(world_state),
                'duration': self._calculate_duration(world_state),
                'affected_regions': self._determine_affected_regions(world_state)
            }
            
            # Record the event generation in the agent's memory
            action_content = {
                'action': {
                    'type': 'GENERATE_EVENT',
                    'details': f"Generated {self.event_type} event with severity {event['severity']:.2f}",
                    'target': 'world',
                    'metadata': event
                }
            }
            
            self._display_communication(
                "assistant",
                action_content,
                "action",
                simplified=True
            )
            
            return event
        return None

    def custom_act(self, world_state: Dict[str, Any] = None, return_actions: bool = False) -> Union[str, List[str]]:
        """Enhanced act method for external event generator."""
        try:
            event = self.generate_event(world_state)
            if event:
                message = (f"Generated {self.event_type} event: severity={event['severity']:.2f}, "
                          f"duration={event['duration']} days, regions={[r.name for r in event['affected_regions']]}")
                return [message] if return_actions else message
            else:
                message = f"No {self.event_type} event generated this time"
                return [message] if return_actions else message
        except Exception as e:
            error_msg = f"Error in External Event Generator {self.name}: {str(e)}"
            logger.error(error_msg)
            return [error_msg] if return_actions else error_msg 