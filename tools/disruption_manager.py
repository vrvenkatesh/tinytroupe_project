from typing import Dict, Any


class DisruptionManager:
    """
    Manages global disruption events and their impact on the supply chain.
    """

    def __init__(self):
        self.active_disruptions = {}

    def add_disruption(self, event_name: str, impact_factors: Dict[str, Any]):
        """
        Adds a new disruption event with specified impact factors.
        """
        self.active_disruptions[event_name] = impact_factors

    def remove_disruption(self, event_name: str):
        """
        Removes an active disruption event.
        """
        if event_name in self.active_disruptions:
            del self.active_disruptions[event_name]

    def get_disruption_impact(self, event_name: str) -> Dict[str, Any]:
        """
        Returns the impact factors of a given disruption event.
        """
        return self.active_disruptions.get(event_name, {})

    def apply_disruptions(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjusts input parameters based on active disruptions.
        """
        for event, impact in self.active_disruptions.items():
            for key, factor in impact.items():
                if key in inputs:
                    inputs[key] *= factor  # Modify inputs dynamically based on disruptions

        return inputs
