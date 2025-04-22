"""Regional Manager Agent for managing regional supply chain operations."""

from typing import Dict, Any, Union, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
import json
import math

from tinytroupe.agent import TinyPerson
from models.enums import Region, OrderStatus
from agents.base import BaseAgent
from models.order import Order

logger = logging.getLogger(__name__)

@dataclass
class RegionalManagerAgent(TinyPerson, BaseAgent):
    """Regional supply chain manager agent."""
    name: str
    config: Dict
    simulation_id: str
    region: Region
    order_batch_size: int = field(init=False)
    order_processing_interval: int = field(init=False)
    regional_demand_weights: Dict[str, float] = field(init=False)
    regional_production_costs: Dict[str, float] = field(init=False)
    pending_orders: List[Order] = field(default_factory=list)
    processed_orders: List[Order] = field(default_factory=list)
    manager_config: Dict = field(init=False)
    local_expertise: float = field(default=0.8)
    cost_sensitivity: float = field(default=0.7)
    adaptability: float = field(default=0.7)

    def __init__(self, name: str, config: Dict[str, Any], simulation_id: str, region: Region):
        """Initialize the regional manager agent."""
        super().__init__(name=name)
        self.config = config
        self.region = region
        self.simulation_id = simulation_id
        
        # Initialize manager configuration
        self.manager_config = self.config['regional_manager']
        
        # Initialize order processing parameters
        self.order_batch_size = self.manager_config['order_batch_size']
        self.order_processing_interval = self.manager_config['order_processing_interval']
        self.regional_demand_weights = self.manager_config['regional_demand_weights']
        self.regional_production_costs = self.manager_config['regional_production_costs']
        
        # Initialize manager attributes from config if provided, otherwise use defaults
        self.local_expertise = self.manager_config.get('local_expertise', self.local_expertise)
        self.cost_sensitivity = self.manager_config.get('cost_sensitivity', self.cost_sensitivity)
        self.adaptability = self.manager_config.get('adaptability', self.adaptability)
        
        # Initialize order lists
        self.pending_orders = []
        self.processed_orders = []

    def receive_order(self, order: Order):
        """Add an order to the pending orders list."""
        self.pending_orders.append(order)

    def process_orders(self, current_time: datetime) -> List[Order]:
        """Process a batch of orders."""
        if not self.pending_orders:
            return []

        # Sort orders by creation time (FIFO)
        self.pending_orders.sort(key=lambda x: x.creation_time)
        
        # Process up to batch_size orders
        to_process = self.pending_orders[:self.order_batch_size]
        processed = []
        
        # Initialize interactions list if not exists
        if not hasattr(self, 'interactions'):
            self.interactions = []
        
        for order in to_process:
            try:
                # Take responsibility for the order if no one has
                if not order.current_handler:
                    order.current_handler = self.name
                
                # Make routing decision and update location
                target_region = self.make_routing_decision(order)
                order.update_location(target_region, "RegionalManager", self.name)
                
                # Process order based on its current status
                if order.status == OrderStatus.NEW:
                    order.update_status(OrderStatus.PRODUCTION, current_time, self.name)
                    self.interactions.append({
                        'type': 'ORDER_STATUS_UPDATE',
                        'timestamp': current_time,
                        'target_agent': self.name,
                        'order_id': order.id,
                        'status': OrderStatus.PRODUCTION,
                        'success': True,
                        'message': f"Order {order.id} moved to production"
                    })
                elif order.status == OrderStatus.PRODUCTION:
                    # Check if production time has elapsed
                    time_in_state = (current_time - order.status_update_time).total_seconds() / (24 * 3600)  # Convert to days
                    if time_in_state >= order.production_time:
                        order.update_status(OrderStatus.READY_FOR_SHIPPING, current_time, self.name)
                        self.interactions.append({
                            'type': 'ORDER_STATUS_UPDATE',
                            'timestamp': current_time,
                            'target_agent': self.name,
                            'order_id': order.id,
                            'status': OrderStatus.READY_FOR_SHIPPING,
                            'success': True,
                            'message': f"Order {order.id} ready for shipping"
                        })
                elif order.status == OrderStatus.READY_FOR_SHIPPING:
                    order.update_status(OrderStatus.IN_TRANSIT, current_time, self.name)
                    self.interactions.append({
                        'type': 'ORDER_STATUS_UPDATE',
                        'timestamp': current_time,
                        'target_agent': self.name,
                        'order_id': order.id,
                        'status': OrderStatus.IN_TRANSIT,
                        'success': True,
                        'message': f"Order {order.id} in transit"
                    })
                elif order.status == OrderStatus.IN_TRANSIT:
                    # Check if transit time has elapsed
                    time_in_state = (current_time - order.status_update_time).total_seconds() / (24 * 3600)  # Convert to days
                    if time_in_state >= (order.transit_time / 24):  # Convert transit_time from hours to days
                        order.update_status(OrderStatus.DELIVERED, current_time, self.name)
                        order.actual_delivery_time = current_time
                        self.interactions.append({
                            'type': 'ORDER_STATUS_UPDATE',
                            'timestamp': current_time,
                            'target_agent': self.name,
                            'order_id': order.id,
                            'status': OrderStatus.DELIVERED,
                            'success': True,
                            'message': f"Order {order.id} delivered"
                        })
                
                processed.append(order)
                self.processed_orders.append(order)
                self.pending_orders.remove(order)
            except (ValueError, PermissionError) as e:
                self.interactions.append({
                    'type': 'ORDER_PROCESSING_ERROR',
                    'timestamp': current_time,
                    'target_agent': self.name,
                    'order_id': order.id,
                    'status': order.status,
                    'success': False,
                    'message': f"Failed to process order {order.id}: {str(e)}"
                })
                print(f"Warning: Failed to process order {order.id}: {str(e)}")
                continue
        
        return processed

    def get_production_cost(self, region: Region) -> float:
        """Get the production cost for a specific region."""
        # Default costs for regions not explicitly specified
        default_costs = {
            'North America': 100,
            'Europe': 120,
            'East Asia': 80,
            'Southeast Asia': 90,
            'South Asia': 85  # Default cost for South Asia
        }
        return self.regional_production_costs.get(region.value, default_costs[region.value])

    def get_demand_weight(self, region: Region) -> float:
        """Get the demand weight for a specific region."""
        # Default weights for regions not explicitly specified
        default_weights = {
            'North America': 0.3,
            'Europe': 0.3,
            'East Asia': 0.2,
            'Southeast Asia': 0.1,
            'South Asia': 0.1  # Default weight for South Asia
        }
        return self.regional_demand_weights.get(region.value, default_weights[region.value])

    def make_routing_decision(self, order: Order) -> Region:
        """Make a routing decision for an order based on costs and demand."""
        best_region = None
        best_score = float('-inf')
        
        # Find min and max costs for normalization
        costs = [self.get_production_cost(r) for r in Region]
        min_cost = min(costs)
        max_cost = max(costs)
        cost_range = max_cost - min_cost
        
        for region in Region:
            # Calculate score based on production cost and demand weight
            cost = self.get_production_cost(region)
            demand = self.get_demand_weight(region)
            
            # Normalize cost to 0-1 range and invert (lower cost = higher score)
            normalized_cost_score = 1 - ((cost - min_cost) / cost_range if cost_range > 0 else 0)
            
            # Cost has 80% weight, demand has 20% weight
            score = (normalized_cost_score * 0.8) + (demand * 0.2)
            
            if score > best_score:
                best_score = score
                best_region = region
        
        return best_region

    def manage_region(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """Manage regional supply chain operations."""
        actions = {
            'supplier_management': self._manage_suppliers(world_state),
            'inventory_management': self._manage_inventory(world_state),
            'transportation_management': self._manage_transportation(world_state),
            'production_management': self._manage_production(world_state),
        }
        return actions

    def _manage_suppliers(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """Manage regional supplier relationships."""
        local_risk = world_state.get(f'{self.region.value}_risk', 0.5)
        local_cost = world_state.get(f'{self.region.value}_cost', 0.5)
        
        # Get values from manager_config
        local_expertise = self.manager_config.get('local_expertise', 0.8)
        cost_sensitivity = self.manager_config.get('cost_sensitivity', 0.7)
        
        return {
            'supplier_engagement': min(1.0, local_expertise * (1.0 - local_risk)),
            'cost_negotiation': max(0.0, 1.0 - local_cost * cost_sensitivity),
        }

    def _manage_inventory(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """Manage regional inventory levels."""
        local_demand = world_state.get(f'{self.region.value}_demand', 0.5)
        local_supply_risk = world_state.get(f'{self.region.value}_supply_risk', 0.5)
        
        return {
            'safety_stock': 1.0 + (local_demand + local_supply_risk) * 0.5,
            'dynamic_adjustment': self.config.get('dynamic_enabled', False),
        }

    def _manage_transportation(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """Manage regional transportation operations."""
        local_infrastructure = world_state.get(f'{self.region.value}_infrastructure', 0.5)
        local_congestion = world_state.get(f'{self.region.value}_congestion', 0.5)
        
        return {
            'mode_selection': {
                'ocean': 0.4 if self.region.value in ['East Asia', 'Southeast Asia'] else 0.2,
                'air': 0.4,
                'ground': 0.4 if self.region.value in ['East Asia', 'Southeast Asia'] else 0.4,
            },
            'route_optimization': min(1.0, self.config['adaptability'] * (1.0 - local_congestion)),
        }

    def _manage_production(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """Manage regional production operations."""
        local_efficiency = world_state.get(f'{self.region.value}_efficiency', 0.5)
        local_flexibility = world_state.get(f'{self.region.value}_flexibility', 0.5)
        
        # Get values from manager_config
        local_expertise = self.manager_config.get('local_expertise', 0.8)
        adaptability = self.manager_config.get('adaptability', 0.7)
        
        return {
            'efficiency_improvement': min(1.0, local_expertise * local_efficiency),
            'flexibility_level': min(1.0, adaptability * local_flexibility),
        } 