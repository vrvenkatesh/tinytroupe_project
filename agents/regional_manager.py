from typing import Dict, Any, Union, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import json

from tinytroupe.agent import TinyPerson
from models.enums import Region, OrderStatus
from agents.base import Agent

logger = logging.getLogger(__name__)

@dataclass
class RegionalManagerAgent(TinyPerson, Agent):
    """Regional supply chain manager agent."""
    region: Region
    config: Dict[str, Any]

    def __init__(self, name: str, config: Dict[str, Any], simulation_id: str, region: Region):
        """Initialize the regional manager agent."""
        super().__init__(name=name)
        self.config = config
        self.region = region
        self.simulation_id = simulation_id  # Store for _post_init

    def _post_init(self, **kwargs):
        """Post-initialization setup."""
        super()._post_init(**kwargs)
        if hasattr(self, 'simulation_id'):
            self._simulation_id = self.simulation_id
            delattr(self, 'simulation_id')

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
        
        return {
            'supplier_engagement': min(1.0, self.config['local_expertise'] * (1.0 - local_risk)),
            'cost_negotiation': max(0.0, 1.0 - local_cost * self.config['cost_sensitivity']),
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
        
        return {
            'efficiency_improvement': min(1.0, self.config['local_expertise'] * local_efficiency),
            'flexibility_level': min(1.0, self.config['adaptability'] * local_flexibility),
        }

    def custom_act(self, world_state: Dict[str, Any] = None, return_actions: bool = False) -> Union[str, List[str]]:
        """Enhanced act method for regional manager with proper interaction recording."""
        if not world_state:
            message = "No world state provided for regional manager to act upon."
            return [message] if return_actions else message
        
        try:
            # Validate required fields
            required_fields = ['active_orders', 'current_datetime', 'production_facilities']
            missing_fields = [field for field in required_fields if field not in world_state]
            if missing_fields:
                raise ValueError(f"Missing required fields in world state: {', '.join(missing_fields)}")
            
            active_orders = world_state['active_orders']
            my_region = self.region
            current_time = world_state['current_datetime']
            
            # Get orders that originate from my region and validate them
            regional_orders = []
            invalid_orders = []
            for order in active_orders:
                if order.source_region == my_region:
                    if order.quantity <= 0:
                        invalid_orders.append((order.id, "Invalid quantity"))
                    elif order.expected_delivery_time < current_time:
                        invalid_orders.append((order.id, "Past due date"))
                    else:
                        regional_orders.append(order)
            
            # Log invalid orders but continue processing
            if invalid_orders:
                logger.warning(f"Regional Manager {self.name} found invalid orders: {invalid_orders}")
            
            # Get available facilities in the region
            facilities = [f for f in world_state['production_facilities']
                         if f.region == my_region]
            
            orders_processed = 0
            messages = []
            
            # Base cognitive state for the regional manager
            base_cognitive_state = {
                'goals': {
                    'primary': 'Optimize regional supply chain operations',
                    'secondary': 'Maintain high facility utilization'
                },
                'attention': {
                    'focus': 'Order processing and facility management',
                    'priority': 'High'
                },
                'emotions': {
                    'stress_level': len(regional_orders) / 100 if regional_orders else 0,
                    'confidence': self.config['local_expertise']
                }
            }
            
            # Process valid orders
            for order in regional_orders:
                if order.status == OrderStatus.NEW:
                    # First, assign to a production facility
                    available_facilities = [
                        f for f in facilities
                        if len(getattr(f, 'current_orders', [])) < getattr(f, 'capacity', 150)
                    ]
                    
                    if available_facilities:
                        facility = available_facilities[0]
                        
                        # Initialize facility's current orders if not exists
                        if not hasattr(facility, 'current_orders'):
                            facility.current_orders = []
                        
                        # Create action content with cognitive state for order assignment
                        action_content = {
                            'action': {
                                'type': 'ASSIGN_ORDER',
                                'details': f"Assigning order {order.id} to facility {facility.name}",
                                'target': facility.name,
                                'metadata': {
                                    'order_id': order.id,
                                    'facility_id': facility.name,
                                    'order_quantity': order.quantity,
                                    'source_region': order.source_region.value,
                                    'destination_region': order.destination_region.value,
                                    'expected_completion': (current_time + timedelta(days=order.production_time)).isoformat()
                                }
                            }
                        }
                        
                        # Store action in memory using TinyPerson's communication mechanism
                        self._display_communication(
                            "assistant",
                            action_content,
                            "action",
                            simplified=True
                        )
                        
                        # Update order status and facility assignment
                        facility.current_orders.append(order.id)
                        order.production_facility = facility.name
                        order.status = OrderStatus.PRODUCTION
                        orders_processed += 1
                        
                        messages.append(f"Assigned order {order.id} to facility {facility.name}")
                    else:
                        # Create action content for failed assignment
                        action_content = {
                            'action': {
                                'type': 'REPORT_CAPACITY_ISSUE',
                                'details': f"No production facilities available for order {order.id}",
                                'target': 'supply_chain',
                                'metadata': {
                                    'order_id': order.id,
                                    'timestamp': current_time.isoformat(),
                                    'reason': 'No available production facilities'
                                }
                            }
                        }
                        
                        self._display_communication(
                            "assistant",
                            action_content,
                            "action",
                            simplified=True
                        )
                        messages.append(f"No production facilities available for order {order.id}")
                
                elif order.status == OrderStatus.PRODUCTION:
                    # Check production progress
                    facility = next((f for f in facilities if order.id in getattr(f, 'current_orders', [])), None)
                    if facility:
                        # Create action content for production monitoring
                        action_content = {
                            'action': {
                                'type': 'MONITOR_PRODUCTION',
                                'details': f"Monitoring production of order {order.id} at facility {facility.name}",
                                'target': facility.name,
                                'metadata': {
                                    'order_id': order.id,
                                    'facility_id': facility.name,
                                    'production_time': order.production_time,
                                    'status': 'in_progress'
                                }
                            }
                        }
                        
                        self._display_communication(
                            "assistant",
                            action_content,
                            "action",
                            simplified=True
                        )
                        messages.append(f"Monitoring production of order {order.id} at facility {facility.name}")
            
            # Add invalid order messages to the output
            for order_id, reason in invalid_orders:
                messages.append(f"Skipped invalid order {order_id}: {reason}")
            
            # Create summary action content
            summary_action = {
                'action': {
                    'type': 'STATUS_SUMMARY',
                    'details': f"Processed {orders_processed} orders in {my_region.value}",
                    'target': 'supply_chain',
                    'metadata': {
                        'orders_processed': orders_processed,
                        'total_orders': len(regional_orders),
                        'region': my_region.value,
                        'timestamp': current_time.isoformat()
                    }
                }
            }
            
            self._display_communication(
                "assistant",
                summary_action,
                "action",
                simplified=True
            )
            messages.append(f"Processed {orders_processed} orders in {my_region.value}")
            
            return messages if return_actions else messages[0] if messages else "No actions taken"
            
        except Exception as e:
            error_msg = f"Error in Regional Manager {self.name}: {str(e)}"
            logger.error(error_msg)
            return [error_msg] if return_actions else error_msg 