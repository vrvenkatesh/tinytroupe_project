from dataclasses import dataclass
from typing import Dict, Any, List, Union

from tinytroupe.agent import TinyPerson
from models.enums import Region
from agents.base import Agent

@dataclass
class SupplierAgent(TinyPerson, Agent):
    """Supplier agent."""
    region: Region
    supplier_type: str  # 'tier1', 'raw_material', 'contract_manufacturer'
    config: Dict[str, Any]

    def __init__(self, name: str, config: Dict[str, Any], simulation_id: str, region: Region, supplier_type: str):
        """Initialize the supplier agent."""
        super().__init__(name=name)
        self.config = config
        self.region = region
        self.supplier_type = supplier_type
        self.simulation_id = simulation_id  # Store for _post_init

    def _post_init(self, **kwargs):
        """Post-initialization setup."""
        super()._post_init(**kwargs)
        if hasattr(self, 'simulation_id'):
            self._simulation_id = self.simulation_id
            delattr(self, 'simulation_id')

    def operate(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """Operate as a supplier in the supply chain."""
        performance = {
            'reliability': self._calculate_reliability(world_state),
            'quality': self._calculate_quality(world_state),
            'cost': self._calculate_cost(world_state),
            'capacity': self._calculate_capacity(world_state),
        }
        
        # Record the operation in TinyPerson's communication
        self._display_communication(
            "assistant",
            {
                "action": {
                    "type": "OPERATE",
                    "details": f"Operating in {self.region.value}",
                    "performance": performance,
                    "target": "supply_chain"  # The target system being operated on
                }
            },
            "action",
            simplified=True
        )
        
        return performance

    def _calculate_reliability(self, world_state: Dict[str, Any]) -> float:
        """Calculate supplier reliability based on various factors."""
        base_reliability = self.config['reliability']
        regional_risk = world_state.get(f'{self.region.value}_risk', 0.5)
        return base_reliability * (1.0 - regional_risk * 0.5)

    def _calculate_quality(self, world_state: Dict[str, Any]) -> float:
        """Calculate supplier quality score."""
        base_quality = self.config['quality_score']
        regional_quality = world_state.get(f'{self.region.value}_quality', 0.5)
        return base_quality * (0.5 + regional_quality * 0.5)

    def _calculate_cost(self, world_state: Dict[str, Any]) -> float:
        """Calculate supplier cost efficiency."""
        base_cost = self.config['cost_efficiency']
        regional_cost = world_state.get(f'{self.region.value}_cost', 0.5)
        return base_cost * (1.0 - regional_cost * 0.3)

    def _calculate_capacity(self, world_state: Dict[str, Any]) -> float:
        """Calculate supplier capacity utilization."""
        base_capacity = 1.0
        regional_capacity = world_state.get(f'{self.region.value}_capacity', 0.5)
        return base_capacity * (0.5 + regional_capacity * 0.5) 