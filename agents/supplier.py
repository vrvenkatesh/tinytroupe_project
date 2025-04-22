from dataclasses import dataclass
from typing import Dict, Any, List, Union

from tinytroupe.agent import TinyPerson
from models.enums import Region
from agents.base import BaseAgent

@dataclass
class SupplierAgent(TinyPerson, BaseAgent):
    """Supplier agent."""
    region: Union[Region, str]  # Can be either Region enum or string
    supplier_type: str  # 'tier1', 'raw_material', 'contract_manufacturer'
    config: Dict[str, Any]

    def __init__(self, name: str, config: Dict[str, Any], simulation_id: str, region: Union[Region, str], supplier_type: str):
        """Initialize the supplier agent."""
        super().__init__(name=name)
        self.config = config
        # Convert string region to Region enum if needed
        if isinstance(region, str):
            try:
                self.region = Region[region.upper().replace(' ', '_')]
            except KeyError:
                # If not a valid enum value, keep as string but normalize format
                self.region = region.replace(' ', '_')
        else:
            self.region = region
        self.supplier_type = supplier_type.lower()  # Normalize to lowercase
        self.simulation_id = simulation_id  # Store for _post_init

        # Handle both config structures (legacy and new)
        if 'supplier' in self.config:
            # Legacy config structure
            self.performance_metrics = {
                'quality': self.config['supplier']['quality'],
                'reliability': self.config['supplier']['reliability'],
                'cost_efficiency': self.config['supplier'].get('cost_efficiency', 0.7),
                'orders_handled': 0,
                'success_rate': 1.0
            }
        else:
            # New config structure
            base_quality = self.config.get('base_quality', {}).get(
                'Tier1' if self.supplier_type == 'tier1' else 'Raw_Material',
                0.85 if self.supplier_type == 'tier1' else 0.75
            )
            region_key = self.region.value if isinstance(self.region, Region) else self.region
            base_reliability = self.config.get('base_reliability', {}).get(
                region_key,
                0.8
            )
            self.performance_metrics = {
                'quality': base_quality,
                'reliability': base_reliability,
                'cost_efficiency': self.config.get('cost_efficiency', 0.7),
                'orders_handled': 0,
                'success_rate': 1.0
            }

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
        
        # Update performance metrics
        self.performance_metrics.update({
            'quality': performance['quality'],
            'reliability': performance['reliability'],
            'cost_efficiency': performance['cost']
        })
        
        # Record the operation in TinyPerson's communication
        region_display = self.region.value if isinstance(self.region, Region) else self.region
        self._display_communication(
            "assistant",
            {
                "action": {
                    "type": "OPERATE",
                    "details": f"Operating in {region_display}",
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
        if 'supplier' in self.config:
            base_reliability = self.config['supplier']['reliability']
        else:
            region_key = self.region.value if isinstance(self.region, Region) else self.region
            base_reliability = self.config.get('base_reliability', {}).get(
                region_key,
                0.8
            )
        region_key = self.region.value if isinstance(self.region, Region) else self.region
        regional_risk = world_state.get(f'{region_key}_risk', 0.5)
        return base_reliability * (1.0 - regional_risk * 0.5)

    def _calculate_quality(self, world_state: Dict[str, Any]) -> float:
        """Calculate supplier quality score."""
        if 'supplier' in self.config:
            base_quality = self.config['supplier']['quality']
        else:
            base_quality = self.config.get('base_quality', {}).get(
                'Tier1' if self.supplier_type == 'tier1' else 'Raw_Material',
                0.85 if self.supplier_type == 'tier1' else 0.75
            )
        region_key = self.region.value if isinstance(self.region, Region) else self.region
        regional_quality = world_state.get(f'{region_key}_quality', 0.5)
        return base_quality * (0.5 + regional_quality * 0.5)

    def _calculate_cost(self, world_state: Dict[str, Any]) -> float:
        """Calculate supplier cost efficiency."""
        if 'supplier' in self.config:
            base_cost = self.config['supplier'].get('cost_efficiency', 0.7)
        else:
            base_cost = self.config.get('cost_efficiency', 0.7)
        region_key = self.region.value if isinstance(self.region, Region) else self.region
        regional_cost = world_state.get(f'{region_key}_cost', 0.5)
        return base_cost * (1.0 - regional_cost * 0.3)

    def _calculate_capacity(self, world_state: Dict[str, Any]) -> float:
        """Calculate supplier capacity utilization."""
        if 'supplier' in self.config:
            region_key = self.region.value if isinstance(self.region, Region) else self.region
            base_capacity = self.config['supplier']['capacity'].get(region_key, 150)
        else:
            base_capacity = 150  # Default capacity
        region_key = self.region.value if isinstance(self.region, Region) else self.region
        regional_capacity = world_state.get(f'{region_key}_capacity', 0.5)
        return (base_capacity / 300) * (0.5 + regional_capacity * 0.5)  # Normalize to 0-1 range

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get the supplier's current performance metrics."""
        return self.performance_metrics.copy()

    def calculate_engagement_level(self, world_state: Dict[str, Any] = None) -> float:
        """Calculate the supplier's engagement level based on various factors.
        
        The engagement level is a measure of how actively involved and responsive
        the supplier is in the supply chain operations. It considers:
        - Supplier type (Tier1 suppliers tend to be more engaged)
        - Regional factors (with stronger regional differentiation)
        - Current performance metrics
        """
        # Base engagement level depends on supplier type
        base_engagement = 0.85 if self.supplier_type == "tier1" else 0.75
        
        # Regional factor (from config or world state)
        region_key = self.region.value if isinstance(self.region, Region) else self.region
        if world_state is not None:
            regional_efficiency = world_state.get(f'{region_key}_efficiency', 0.5)
        else:
            # Get regional engagement factor with stronger regional differentiation
            regional_efficiency = self.config.get('engagement_factors', {}).get(
                region_key,
                0.8 if region_key.lower().replace('_', ' ') == 'north america' else 0.6  # Higher default for NA
            )
        
        # Performance factor (average of quality and reliability)
        performance_factor = (self.performance_metrics['quality'] + 
                            self.performance_metrics['reliability']) / 2
        
        # Calculate final engagement level with stronger regional weight
        engagement_level = (
            base_engagement * 0.3 +  # 30% weight on base engagement
            regional_efficiency * 0.5 +  # 50% weight on regional efficiency (increased from 30%)
            performance_factor * 0.2  # 20% weight on performance (decreased from 30%)
        )
        
        return min(1.0, max(0.0, engagement_level))  # Ensure result is between 0 and 1 