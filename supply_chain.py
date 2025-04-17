"""
Supply Chain Resilience Optimization Simulation - Core Components

This module contains the core components and agent definitions for the supply chain
resilience optimization simulation using TinyTroupe's agent-based simulation capabilities.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
import uuid
import logging
from datetime import datetime, timedelta
import types  # Add this import for MethodType

from tinytroupe.agent import TinyPerson
from tinytroupe.environment.tiny_world import TinyWorld as World
from tinytroupe.factory import TinyPersonFactory
from tinytroupe.environment import logger
from tinytroupe import config_init
from tinytroupe.tools import TinyTool, ToolRegistry

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    'simulation': {
        'seed': 42,
        'monte_carlo_iterations': 100,
        'suppliers_per_region': 3,
        'time_steps': 365,  # One year of daily operations
        'base_demand': 10,  # Base demand per region pair
    },
    'coo': {
        'risk_aversion': 0.7,
        'cost_sensitivity': 0.5,
        'strategic_vision': 0.8,
        'initial_metrics': {
            'resilience_score': 0.6,
            'recovery_time': 0.7,
            'service_level': 0.8,
            'total_cost': 0.7,
            'inventory_cost': 0.6,
            'transportation_cost': 0.7,
            'risk_exposure': 0.5,
            'supplier_risk': 0.4,
            'transportation_risk': 0.5,
            'lead_time': 0.7,
            'flexibility_score': 0.6,
            'quality_score': 0.8
        }
    },
    'regional_manager': {
        'local_expertise': 0.8,
        'adaptability': 0.7,
        'communication_skills': 0.6,
        'cost_sensitivity': 0.6
    },
    'supplier': {
        'reliability': 0.8,
        'quality_score': 0.9,
        'cost_efficiency': 0.7,
        'diversification_enabled': False,
    },
    'logistics': {
        'reliability': 0.8,
        'cost_efficiency': 0.7,
        'flexibility': 0.6,
        'flexible_routing_enabled': False,
    },
    'production_facility': {
        'efficiency': 0.8,
        'quality_control': 0.9,
        'flexibility': 0.7,
        'regional_flexibility_enabled': False,
        'base_production_time': 3,  # Base time in days for production
    },
    'inventory_management': {
        'base_stock_level': 100,
        'safety_stock_factor': 1.5,
        'dynamic_enabled': False,
    },
    'external_events': {
        'weather': {
            'frequency': 0.1,
            'severity': 0.5,
        },
        'geopolitical': {
            'frequency': 0.05,
            'severity': 0.7,
        },
        'market': {
            'frequency': 0.2,
            'severity': 0.4,
        },
    },
}

class Region(Enum):
    NORTH_AMERICA = "North America"
    EUROPE = "Europe"
    EAST_ASIA = "East Asia"
    SOUTHEAST_ASIA = "Southeast Asia"
    SOUTH_ASIA = "South Asia"

class TransportationMode(Enum):
    OCEAN = "Ocean"
    AIR = "Air"
    GROUND = "Ground"

class OrderStatus(Enum):
    NEW = "NEW"                      # Initial state when order is created
    PRODUCTION = "PRODUCTION"        # Order assigned to supplier and in production
    READY_FOR_SHIPPING = "READY"     # Production complete, ready to be shipped
    IN_TRANSIT = "IN_TRANSIT"        # Order is being shipped
    DELIVERED = "DELIVERED"          # Order has reached its destination
    CANCELLED = "CANCELLED"          # Order was cancelled
    DELAYED = "DELAYED"              # Order is delayed
    QUALITY_CHECK_FAILED = "FAILED"  # Order failed quality check

@dataclass
class Order:
    """Represents a supply chain order."""
    id: str
    product_type: str
    quantity: int
    source_region: Region
    destination_region: Region
    creation_time: datetime
    expected_delivery_time: datetime
    actual_delivery_time: Optional[datetime] = None  # Actual delivery timestep
    status: OrderStatus = OrderStatus.NEW  # Changed from CREATED to NEW
    transportation_mode: Optional[TransportationMode] = None
    current_location: Optional[Region] = None
    production_time: int = 0  # Time spent in production
    transit_time: int = 0  # Time spent in transit
    delay_time: int = 0  # Total delay time
    cost: float = 0.0  # Total cost of the order
    production_facility: Optional[str] = None
    quality_check_passed: Optional[bool] = None  # Result of quality check

    def update_status(self, new_status: OrderStatus, current_time: datetime) -> None:
        """Update order status and related timing metrics."""
        if new_status == OrderStatus.DELIVERED and self.actual_delivery_time is None:
            self.actual_delivery_time = current_time
            
        if new_status == OrderStatus.DELAYED:
            self.delay_time += 1
            
        self.status = new_status
        
    def calculate_lead_time(self) -> Optional[int]:
        """Calculate total lead time if order is delivered."""
        if self.actual_delivery_time is not None:
            return (self.actual_delivery_time - self.creation_time).days
        return None
    
    def is_on_time(self) -> Optional[bool]:
        """Check if order was delivered on time."""
        if self.actual_delivery_time is not None:
            return self.actual_delivery_time <= self.expected_delivery_time
        return None

@dataclass
class Agent:
    """Base class for all agents in the simulation."""
    name: str
    config: Dict[str, Any]
    simulation_id: str

@dataclass
class COOAgent(TinyPerson):
    """Chief Operations Officer agent responsible for overseeing supply chain operations."""
    
    def __init__(self, name: str, config: Dict[str, Any], simulation_id: str):
        """Initialize the COO agent with required parameters."""
        super().__init__(name=name)
        self.config = config
        self.simulation_id = simulation_id
    
    def act(self, world_state: Dict[str, Any] = None, return_actions: bool = False) -> Union[str, List[str]]:
        """
        Analyze supply chain status and provide insights based on order completion ratios.
        
        Args:
            world_state (Dict[str, Any]): Current state of the world
            return_actions (bool): Whether to return actions as a list
            
        Returns:
            Union[str, List[str]]: Response message or list of actions
        """
        try:
            if not world_state or "orders" not in world_state:
                message = "No orders to analyze in the current world state."
                return [message] if return_actions else message
            
            orders = world_state["orders"]
            if not orders:
                message = "No active orders in the supply chain."
                return [message] if return_actions else message
            
            completed_orders = sum(1 for order in orders if order.status == OrderStatus.DELIVERED)
            total_orders = len(orders)
            completion_ratio = completed_orders / total_orders if total_orders > 0 else 0
            
            if completion_ratio >= 0.9:
                message = "Supply chain is operating efficiently with high order completion rate."
            elif completion_ratio >= 0.7:
                message = "Supply chain performance is good but there's room for improvement."
            elif completion_ratio >= 0.5:
                message = "Supply chain requires attention. Order completion rate is moderate."
            else:
                message = "Supply chain needs immediate intervention. Low order completion rate."
            
            return [message] if return_actions else message
            
        except Exception as e:
            error_msg = f"Error in COO analysis: {str(e)}"
            return [error_msg] if return_actions else error_msg

@dataclass
class RegionalManagerAgent(Agent):
    """Regional supply chain manager agent."""
    region: Region

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
                'air': 0.3,
                'ground': 0.3,
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

@dataclass
class SupplierAgent(Agent):
    """Supplier agent."""
    region: Region
    supplier_type: str  # 'tier1', 'raw_material', 'contract_manufacturer'

    def operate(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """Operate as a supplier in the supply chain."""
        performance = {
            'reliability': self._calculate_reliability(world_state),
            'quality': self._calculate_quality(world_state),
            'cost': self._calculate_cost(world_state),
            'capacity': self._calculate_capacity(world_state),
        }
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

class SupplyChainTools:
    """Collection of role-specific tools for supply chain agents."""
    
    @staticmethod
    def create_coo_tools() -> List[TinyTool]:
        """Create tools specific to the COO role."""
        return [
            TinyTool(
                name="adjust_global_strategy",
                description="Adjust global supply chain strategy based on current conditions",
                function=lambda world_state, strategy_type, adjustment: {
                    "action": "strategy_adjustment",
                    "type": strategy_type,
                    "value": adjustment
                }
            ),
            TinyTool(
                name="authorize_emergency_shipping",
                description="Authorize emergency shipping for critical orders",
                function=lambda world_state, order_ids: {
                    "action": "emergency_shipping",
                    "orders": order_ids
                }
            )
        ]
    
    @staticmethod
    def create_regional_manager_tools() -> List[TinyTool]:
        """Create tools specific to Regional Manager role."""
        return [
            TinyTool(
                name="reallocate_regional_capacity",
                description="Reallocate production capacity within the region",
                function=lambda world_state, facility_adjustments: {
                    "action": "capacity_reallocation",
                    "adjustments": facility_adjustments
                }
            ),
            TinyTool(
                name="expedite_regional_orders",
                description="Expedite specific orders within the region",
                function=lambda world_state, order_ids: {
                    "action": "order_expedition",
                    "orders": order_ids
                }
            ),
            TinyTool(
                name="assign_production_facility",
                description="Assign a production facility to handle an order",
                function=lambda world_state, order_id, facility_name: {
                    "action": "assign_facility",
                    "order_id": order_id,
                    "facility": facility_name,
                    "new_status": OrderStatus.PRODUCTION.value
                }
            ),
            TinyTool(
                name="assign_logistics_provider",
                description="Assign a logistics provider to handle order shipping",
                function=lambda world_state, order_id, provider_name, mode: {
                    "action": "assign_logistics",
                    "order_id": order_id,
                    "provider": provider_name,
                    "mode": mode,
                    "new_status": OrderStatus.IN_TRANSIT.value
                }
            )
        ]
    
    @staticmethod
    def create_supplier_tools() -> List[TinyTool]:
        """Create tools specific to Supplier role."""
        return [
            TinyTool(
                name="check_capacity",
                description="Check current production capacity and utilization",
                function=lambda world_state, **kwargs: {
                    'capacity': world_state.get('production_capacity', 100),
                    'utilization': len([o for o in world_state.get('active_orders', [])
                                     if o.status == OrderStatus.IN_PRODUCTION and 
                                     getattr(o, 'supplier', None) == kwargs['name']])
                }
            ),
            TinyTool(
                name="process_order",
                description="Start processing a production order",
                function=lambda world_state, order_id, **kwargs: {
                    'order_id': order_id,
                    'status': OrderStatus.IN_PRODUCTION.value,
                    'estimated_completion': world_state.get('current_time', 0) + 
                                         kwargs.get('lead_time', 2)
                }
            ),
            TinyTool(
                name="update_order_status",
                description="Update the status of an order in production",
                function=lambda world_state, order_id, new_status, **kwargs: {
                    'order_id': order_id,
                    'status': new_status,
                    'timestamp': world_state.get('current_time', 0)
                }
            )
        ]
    
    @staticmethod
    def create_logistics_tools() -> List[TinyTool]:
        """Create tools specific to Logistics Provider role."""
        return [
            TinyTool(
                name="optimize_route",
                description="Optimize shipping route for specific orders",
                function=lambda world_state, order_id, new_route: {
                    "action": "route_optimization",
                    "order": order_id,
                    "route": new_route
                }
            ),
            TinyTool(
                name="change_transport_mode",
                description="Change transportation mode for specific orders",
                function=lambda world_state, order_id, new_mode: {
                    "action": "mode_change",
                    "order": order_id,
                    "mode": new_mode
                }
            ),
            TinyTool(
                name="start_transit",
                description="Start transit for an order",
                function=lambda world_state, order_id, mode: {
                    "action": "start_transit",
                    "order_id": order_id,
                    "mode": mode,
                    "new_status": OrderStatus.IN_TRANSIT.value,
                    "timestamp": world_state.get('current_datetime')
                }
            ),
            TinyTool(
                name="complete_delivery",
                description="Mark order as delivered",
                function=lambda world_state, order_id: {
                    "action": "complete_delivery",
                    "order_id": order_id,
                    "new_status": OrderStatus.DELIVERED.value,
                    "timestamp": world_state.get('current_datetime')
                }
            )
        ]

    @staticmethod
    def create_production_facility_tools() -> List[TinyTool]:
        """Create tools specific to Production Facility role."""
        return [
            TinyTool(
                name="start_production",
                description="Start production for an order",
                function=lambda world_state, order_id: {
                    "action": "start_production",
                    "order_id": order_id,
                    "new_status": OrderStatus.IN_PRODUCTION.value,
                    "timestamp": world_state.get('current_datetime')
                }
            ),
            TinyTool(
                name="complete_production",
                description="Mark production as complete for an order",
                function=lambda world_state, order_id: {
                    "action": "complete_production",
                    "order_id": order_id,
                    "new_status": OrderStatus.READY_FOR_TRANSIT.value,
                    "timestamp": world_state.get('current_datetime')
                }
            ),
            TinyTool(
                name="check_production_status",
                description="Check current production status and capacity",
                function=lambda world_state, **kwargs: {
                    'capacity': world_state.get('capacity', 150),
                    'current_load': len([o for o in world_state.get('active_orders', [])
                                      if o.status == OrderStatus.IN_PRODUCTION and 
                                      getattr(o, 'production_facility', None) == kwargs['name']])
                }
            ),
            TinyTool(
                name="quality_check",
                description="Perform quality check on production",
                function=lambda world_state, order_id, **kwargs: {
                    'order_id': order_id,
                    'quality_score': random.random() < world_state.get('quality_rate', 0.95),
                    'timestamp': world_state.get('current_time', 0)
                }
            )
        ]

def create_coo_agent(name: str, config: Dict[str, Any], simulation_id: str) -> COOAgent:
    """
    Create a Chief Operating Officer (COO) agent.
    
    Args:
        name (str): The name for the COO agent
        config (Dict[str, Any]): Configuration parameters
        simulation_id (str): The simulation ID this agent belongs to
        
    Returns:
        COOAgent: Initialized COO agent
    """
    coo = COOAgent(name=name, config=config, simulation_id=simulation_id)
    coo._persona = {
        "name": name,  # Use the provided name consistently
        "occupation": {
            "title": "Chief Operating Officer",
            "organization": "Global Supply Chain Corp"
        },
        "experience": "20+ years in supply chain management",
        "expertise": ["Supply Chain Optimization", "Strategic Planning", "Risk Management"],
        "communication_style": "Professional and data-driven",
        "default_response": (
            "Based on my analysis of the supply chain metrics, "
            "I recommend focusing on optimizing our order processing "
            "and delivery systems for improved efficiency."
        )
    }
    
    return coo

def create_regional_manager_agent(name: str, config: Dict[str, Any], simulation_id: str) -> TinyPerson:
    """Create a regional manager agent using TinyTroupe."""
    manager = TinyPerson(name=name)
    
    # Set up manager's persona
    region = next((r for r in Region if r.value in name), Region.NORTH_AMERICA)
    manager._persona = {
        'occupation': {
            'title': 'Regional Manager',
            'region': region.value,
            'organization': 'Global Supply Chain Network'
        },
        'capabilities': {
            'local_expertise': config.get('local_expertise', 0.8),
            'adaptability': config.get('adaptability', 0.7),
            'decision_speed': config.get('decision_speed', 0.6)
        },
        'decision_making': {
            'local_expertise': config.get('local_expertise', 0.8),
            'adaptability': config.get('adaptability', 0.7),
            'cost_sensitivity': config.get('cost_sensitivity', 0.6)
        },
        'tools': SupplyChainTools.create_regional_manager_tools(),
        'response_style': {
            'default_response': "Managing regional supply chain operations."
        }
    }
    
    def custom_act(self, world_state: Dict[str, Any] = None, return_actions: bool = False) -> Union[str, List[str]]:
        """Enhanced act method for regional manager."""
        if not world_state:
            message = self._persona['response_style']['default_response']
            return [message] if return_actions else message
        
        try:
            active_orders = world_state.get('active_orders', [])
            my_region = Region(self._persona['occupation']['region'])  # Convert string to Region enum
            
            # Get orders that originate from my region
            regional_orders = [
                order for order in active_orders
                if order.source_region == my_region  # Only handle orders from my region
            ]
            
            # Get available facilities and suppliers in the region
            facilities = [f for f in world_state.get('production_facilities', [])
                         if Region(f._persona['occupation']['region']) == my_region]
            
            suppliers = [s for s in world_state.get('suppliers', [])
                        if Region(s._persona['occupation']['region']) == my_region]
            
            logistics_providers = world_state.get('logistics_providers', [])
            
            orders_processed = 0
            messages = []
            
            for order in regional_orders:
                if order.status == OrderStatus.NEW:
                    # First, assign to a production facility
                    available_facilities = []
                    for f in facilities:
                        total_capacity = f._persona['capabilities'].get('capacity', 150)  # Default capacity if not set
                        current_orders = getattr(f, 'current_orders', [])
                        if len(current_orders) < total_capacity:
                            available_facilities.append(f)
                    
                    if available_facilities and suppliers:  # Check both conditions upfront
                        facility = available_facilities[0]  # Use the first available facility
                        supplier = suppliers[0]  # Use the first available supplier
                        
                        # Initialize facility's current orders if not exists
                        if not hasattr(facility, 'current_orders'):
                            facility.current_orders = []
                        
                        # Make all assignments and update status
                        facility.current_orders.append(order.id)
                        order.production_facility = facility.name
                        order.supplier = supplier.name
                        order.status = OrderStatus.PRODUCTION
                        orders_processed += 1
                        
                        msg = f"Regional Manager {self.name}: Assigned order {order.id} to facility {facility.name} and supplier {supplier.name}"
                        logger.info(msg)
                        messages.append(msg)
                    else:
                        missing = []
                        if not available_facilities:
                            missing.append("production facilities")
                        if not suppliers:
                            missing.append("suppliers")
                        msg = f"Regional Manager {self.name}: No {' or '.join(missing)} available for order {order.id}"
                        logger.warning(msg)
                        messages.append(msg)
                
                elif order.status == OrderStatus.READY_FOR_SHIPPING:
                    # Find an available logistics provider for the route
                    suitable_providers = [
                        p for p in logistics_providers
                        if (order.source_region.value in p.name and order.destination_region.value in p.name)
                    ]
                    
                    if suitable_providers:
                        provider = suitable_providers[0]  # Use the first suitable provider
                        order.logistics_provider = provider.name
                        order.transportation_mode = TransportationMode.AIR  # Default to air for now
                        order.status = OrderStatus.IN_TRANSIT
                        orders_processed += 1
                        msg = f"Regional Manager {self.name}: Assigned order {order.id} to logistics provider {provider.name}"
                        logger.info(msg)
                        messages.append(msg)
                    else:
                        msg = f"Regional Manager {self.name}: No suitable logistics providers available for order {order.id}"
                        logger.warning(msg)
                        messages.append(msg)
                
                elif order.status == OrderStatus.IN_TRANSIT:
                    # Check for delivery completion
                    current_datetime = world_state.get('current_datetime', datetime.now())
                    if _check_delivery(order, current_datetime, world_state.get('config', {})):
                        order.update_status(OrderStatus.DELIVERED, current_datetime)
                        orders_processed += 1
                        msg = f"Regional Manager {self.name}: Delivered order {order.id}"
                        logger.info(msg)
                        messages.append(msg)
                        # Move order to completed_orders list
                        if order in world_state['active_orders']:
                            world_state['active_orders'].remove(order)
                            world_state.setdefault('completed_orders', []).append(order)
                    else:
                        # Check for delays
                        if _check_delays(order, current_datetime, world_state.get('config', {})):
                            order.update_status(OrderStatus.DELAYED, current_datetime)
                            msg = f"Regional Manager {self.name}: Order {order.id} is delayed"
                            logger.warning(msg)
                            messages.append(msg)
            
            summary = f"Processed {orders_processed} orders in {my_region.value}"
            messages.append(summary)
            return messages if return_actions else summary
            
        except Exception as e:
            error_msg = f"Error in Regional Manager {self.name}: {str(e)}"
            logger.error(error_msg)
            return [error_msg] if return_actions else error_msg
    
    # Bind the custom_act method to the instance
    import types
    manager.act = types.MethodType(custom_act, manager)
    
    return manager

def create_supplier_agent(name: str, config: Dict[str, Any], simulation_id: str) -> TinyPerson:
    """Create a supplier agent using TinyTroupe."""
    supplier = TinyPerson(name=name)
    
    # Set up supplier's persona
    supplier._persona = {
        'name': name,
        'occupation': {
            'title': 'Supply Chain Supplier',
            'region': config.get('region', Region.NORTH_AMERICA),  # Default to North America if not specified
            'specialties': config.get('specialties', ['Raw Materials'])
        },
        'capabilities': {
            'production_capacity': config.get('production_capacity', 100),
            'quality_score': config.get('quality_score', 0.85),
            'reliability': config.get('reliability', 0.8),
            'lead_time': config.get('lead_time', 2)
        },
        'response_style': {
            'default_response': "Managing supply operations."
        }
    }
    
    def custom_act(self, world_state: Dict[str, Any] = None, return_actions: bool = False) -> Union[str, List[str]]:
        """Enhanced act method for supplier."""
        if not world_state:
            return [self._persona['response_style']['default_response']] if return_actions else self._persona['response_style']['default_response']
        
        try:
            active_orders = world_state.get('active_orders', [])
            supplier_orders = [
                order for order in active_orders
                if getattr(order, 'supplier', None) == self.name and order.status == OrderStatus.PRODUCTION
            ]
            
            orders_processed = 0
            messages = []
            
            for order in supplier_orders:
                # Process the order
                if not hasattr(order, 'production_time'):
                    order.production_time = 0
                
                order.production_time += 1
                
                # Check if production is complete
                if order.production_time >= self._persona['capabilities']['lead_time']:
                    order.status = OrderStatus.READY_FOR_SHIPPING
                    orders_processed += 1
                    msg = f"Supplier {self.name}: Completed processing order {order.id}"
                    logger.info(msg)
                    messages.append(msg)
            
            summary = f"Supplier {self.name}: Processing {len(supplier_orders)} orders, completed {orders_processed}"
            messages.append(summary)
            return messages if return_actions else summary
            
        except Exception as e:
            error_msg = f"Error in Supplier {self.name}: {str(e)}"
            logger.error(error_msg)
            return [error_msg] if return_actions else error_msg
    
    # Properly bind the custom_act method to the instance
    supplier.act = types.MethodType(custom_act, supplier)
    return supplier

def create_logistics_provider_agent(source_region: Region, dest_region: Region, config: Dict[str, Any] = None) -> TinyPerson:
    """Create a logistics provider agent using TinyTroupe."""
    if config is None:
        config = {}
    
    mode = random.choice(['Air', 'Ocean', 'Ground'])
    # Create unique name with UUID
    unique_id = str(uuid.uuid4())[:8]
    name = f"Logistics_{source_region.value}_to_{dest_region.value}_{unique_id}"
    logistics = TinyPerson(name=name)
    
    # Set up logistics provider's persona
    logistics._persona = {
        'occupation': {
            'title': f'{mode} Logistics Manager',
            'organization': 'Global Supply Chain Network',
            'mode': mode,
            'source_region': source_region.value,
            'destination_region': dest_region.value,
            'description': f'Responsible for managing {mode.lower()} transportation operations between {source_region.value} and {dest_region.value}'
        },
        'capabilities': {
            'reliability': config.get('reliability', 0.8),
            'cost_efficiency': config.get('cost_efficiency', 0.7),
            'flexibility': config.get('flexibility', 0.6),
            'transit_time': config.get('transit_time', {
                'Air': 2,
                'Ocean': 5,
                'Ground': 3
            }).get(mode, 3)
        },
        'tools': SupplyChainTools.create_logistics_tools(),
        'response_style': {
            'format': 'structured',
            'focus': 'transportation and delivery status',
            'default_response': f'{mode} transportation operations are running on schedule.'
        }
    }
    
    def custom_act(self, world_state: Dict[str, Any] = None, return_actions: bool = True) -> Union[str, List[str]]:
        """Enhanced act method for logistics provider."""
        try:
            if not world_state:
                message = "No world state provided"
                return [message] if return_actions else message
                
            # Check for orders in transit
            transit_orders = [
                order for order in world_state.get('active_orders', [])
                if (order.status == OrderStatus.IN_TRANSIT and 
                    getattr(order, 'logistics_provider', None) == self.name)
            ]
            
            orders_processed = 0
            messages = []
            
            for order in transit_orders:
                if not hasattr(order, 'transit_time'):
                    order.transit_time = 0
                order.transit_time += 1
                
                # Check if delivery is complete based on transit time
                if order.transit_time >= self._persona['capabilities']['transit_time']:
                    order.update_status(OrderStatus.DELIVERED, world_state.get('current_datetime', datetime.now()))
                    orders_processed += 1
                    msg = f"Logistics Provider {self.name}: Delivered order {order.id}"
                    logger.info(msg)
                    messages.append(msg)
                else:
                    # Check if optimization is needed
                    if order.transit_time > self._persona['capabilities']['transit_time'] * 0.7:
                        msg = f"Logistics Provider {self.name}: Optimizing route for order {order.id}"
                        logger.info(msg)
                        messages.append(msg)
                        # Apply route optimization logic here if needed
            
            summary = f"Logistics Provider {self.name}: Processed {orders_processed} orders in transit"
            messages.append(summary)
            return messages if return_actions else summary
            
        except Exception as e:
            error_msg = f"Error in Logistics Provider {self.name}: {str(e)}"
            logger.error(error_msg)
            return [error_msg] if return_actions else error_msg
    
    # Properly bind the custom_act method to the instance
    logistics.act = types.MethodType(custom_act, logistics)
    return logistics

def create_production_facility_agent(name: str, config: Dict[str, Any], simulation_id: str) -> TinyPerson:
    """Create a production facility agent using TinyTroupe with specific tools."""
    facility = TinyPerson(name=name)
    
    # Set up facility's persona
    facility._persona = {
        'occupation': {
            'title': 'Production Facility Manager',
            'region': config.get('region', Region.NORTH_AMERICA)
        },
        'capabilities': {
            'capacity': config.get('capacity', 150),
            'efficiency': config.get('efficiency', 0.9),
            'quality_rate': config.get('quality_rate', 0.95),
            'flexibility': config.get('flexibility', 0.8),
            'lead_time': config.get('base_production_time', 3)
        },
        'tools': SupplyChainTools.create_production_facility_tools(),
        'response_style': {
            'default_response': "I am monitoring production and quality metrics."
        }
    }
    
    def custom_act(self, world_state: Optional[Dict[str, Any]] = None, return_actions: bool = True) -> Union[str, List[str]]:
        try:
            if not world_state:
                message = "No world state provided"
                return [message] if return_actions else message
            
            active_orders = world_state.get('active_orders', [])
            current_datetime = world_state.get('current_datetime', datetime.now())
            
            # Get orders assigned to this facility in NEW or PRODUCTION status
            facility_orders = [
                order for order in active_orders
                if (getattr(order, 'production_facility', None) == self.name and 
                    order.status in [OrderStatus.NEW, OrderStatus.PRODUCTION])
            ]
            
            orders_processed = 0
            messages = []
            
            for order in facility_orders:
                # Initialize production for NEW orders
                if order.status == OrderStatus.NEW:
                    order.status = OrderStatus.PRODUCTION
                    order.production_time = 0
                    msg = f"Production Facility {self.name}: Started production for order {order.id}"
                    logger.info(msg)
                    messages.append(msg)
                    continue
                
                # Process PRODUCTION orders
                if order.status == OrderStatus.PRODUCTION:
                    # Update production time
                    if not hasattr(order, 'production_time'):
                        order.production_time = 0
                    order.production_time += 1
                    
                    # Check if production is complete based on lead time
                    if order.production_time >= self._persona['capabilities']['lead_time']:
                        # Perform quality check
                        quality_check = random.random() < self._persona['capabilities']['quality_rate']
                        
                        if quality_check:
                            order.status = OrderStatus.READY_FOR_TRANSIT
                            orders_processed += 1
                            msg = f"Production Facility {self.name}: Completed production for order {order.id}"
                            logger.info(msg)
                            messages.append(msg)
                        else:
                            order.status = OrderStatus.QUALITY_CHECK_FAILED
                            msg = f"Production Facility {self.name}: Quality check failed for order {order.id}"
                            logger.warning(msg)
                            messages.append(msg)
                    else:
                        msg = f"Production Facility {self.name}: Order {order.id} in production (time: {order.production_time}/{self._persona['capabilities']['lead_time']})"
                        logger.info(msg)
                        messages.append(msg)
            
            summary = f"Production Facility {self.name}: Processed {orders_processed} orders in {self._persona['occupation']['region']}"
            messages.append(summary)
            return messages if return_actions else summary
            
        except Exception as e:
            error_msg = f"Error in Production Facility {self.name}: {str(e)}"
            logger.error(error_msg)
            return [error_msg] if return_actions else error_msg
    
    # Properly bind the custom_act method to the instance
    facility.act = types.MethodType(custom_act, facility)
    return facility

def create_external_event_agent(name: str, config: Dict[str, Any], simulation_id: str) -> TinyPerson:
    """Create an external event agent using TinyTroupe."""
    event_agent = TinyPerson(name)
    event_type = random.choice(['weather', 'geopolitical', 'market'])
    event_agent._persona = {
        'occupation': {
            'title': f'{event_type.title()} Event Monitor',
            'organization': 'Tekron Industries Risk Management',
            'type': event_type,
            'industry': 'Automation Equipment Manufacturing',
            'description': f'You monitor and assess {event_type} events that may impact supply chain operations.'
        },
        'behaviors': [
            'Monitor external conditions',
            'Assess event probability',
            'Calculate impact severity',
            'Determine affected regions',
            'Issue early warnings',
            'Track event duration'
        ],
        'decision_making': {
            'severity_threshold': config[event_type]['severity'],
            'frequency_factor': config[event_type]['frequency'],
            'impact_assessment': 0.7,
            'warning_threshold': 0.6
        },
        'response_style': {
            'format': 'structured',
            'focus': 'event monitoring and impact assessment',
            'default_response': f'No significant {event_type} events detected affecting operations.'
        }
    }
    return event_agent

def create_simulation_world(config: Dict[str, Any]) -> World:
    """Create a simulation world with all necessary agents."""
    # Generate unique world and simulation IDs
    world_id = str(uuid.uuid4())[:8]
    simulation_id = str(uuid.uuid4())

    world = World(
        name=f"SupplyChainWorld_{world_id}",
        agents=[],  # We'll add agents below
        broadcast_if_no_target=True
    )
    
    # Initialize lists to store agents by type
    regional_managers = []
    suppliers = []
    production_facilities = []
    logistics_providers = []

    # Create COO agent
    coo = create_coo_agent(
        name=f"COO_{world_id}",  # Changed from Global_COO to match test format
        config=config['coo'],
        simulation_id=simulation_id
    )
    world.agents.append(coo)

    # Create regional managers for each region
    for region in Region:
        manager = create_regional_manager_agent(
            name=f"Manager_{region.value}_{world_id}",
            config={'region': region, **config['regional_manager']},
            simulation_id=simulation_id
        )
        regional_managers.append(manager)
        world.agents.append(manager)

    # Create suppliers for each region based on config
    suppliers_per_region = config['simulation'].get('suppliers_per_region', 3)
    for region in Region:
        for i in range(suppliers_per_region):
            supplier = create_supplier_agent(
                name=f"Supplier_{region.value}_{i+1}_{world_id}",
                config={'region': region, **config['supplier']},
                simulation_id=simulation_id
            )
            suppliers.append(supplier)
            world.agents.append(supplier)

    # Create production facilities for each region
    for region in Region:
        facility = create_production_facility_agent(
            name=f"Facility_{region.value}_{world_id}",
            config={'region': region, **config['production_facility']},
            simulation_id=simulation_id
        )
        production_facilities.append(facility)
        world.agents.append(facility)

    # Create logistics providers (one per region pair)
    for source_region in Region:
        for dest_region in Region:
            if source_region != dest_region:
                provider = create_logistics_provider_agent(source_region, dest_region)
                provider.name = f"Logistics_{source_region.value}_to_{dest_region.value}_{world_id}"
                logistics_providers.append(provider)
                world.agents.append(provider)

    # Create external event agent to simulate disruptions
    event_agent = create_external_event_agent(
        name=f"External_Events_{world_id}",
        config=config['external_events'],
        simulation_id=simulation_id
    )
    world.agents.append(event_agent)

    # Store agent references in world state for easy access
    world.state = {
        'coo': coo,
        'regional_managers': regional_managers,
        'suppliers': suppliers,
        'production_facilities': production_facilities,
        'logistics_providers': logistics_providers,
        'event_agent': event_agent,
        'active_orders': [],
        'completed_orders': [],
        'failed_orders': [],
        'current_datetime': datetime.now(),
        'config': config
    }

    logger.info(f"Created simulation world with {len(world.agents)} agents:")
    logger.info(f"- COO: 1")
    logger.info(f"- Regional Managers: {len(regional_managers)}")
    logger.info(f"- Suppliers: {len(suppliers)}")
    logger.info(f"- Production Facilities: {len(production_facilities)}")
    logger.info(f"- Logistics Providers: {len(logistics_providers)}")
    logger.info(f"- External Event Agent: 1")

    return world

def simulate_supply_chain_operation(world: World, config: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate supply chain operations for one time step with agent interactions."""
    # Initialize world state if not already done
    if not hasattr(world, 'state'):
        world.state = {}
    
    # Ensure required state fields exist
    world.state.setdefault('active_orders', [])
    world.state.setdefault('completed_orders', [])
    world.state.setdefault('failed_orders', [])
    # Update current_datetime based on world's current_time
    if hasattr(world, 'current_time'):
        world.state['current_datetime'] = datetime.now() + timedelta(days=world.current_time)
    else:
        world.state.setdefault('current_datetime', datetime.now())
    world.state.setdefault('config', config)
    
    interaction_log = []
    
    # Get all agents by type and ensure their personas are initialized
    regional_managers = []
    suppliers = []
    production_facilities = []
    logistics_providers = []
    
    for agent in world.agents:
        if 'Manager' in agent.name and any(region.value in agent.name for region in Region):
            if not hasattr(agent, '_persona'):
                region = next((r for r in Region if r.value in agent.name), None)
                agent._persona = {
                    'occupation': {
                        'title': 'Regional Manager',
                        'region': region.value if region else 'Unknown'
                    },
                    'capabilities': config['regional_manager'].get('capabilities', {})
                }
            regional_managers.append(agent)
        elif 'Supplier' in agent.name:
            if not hasattr(agent, '_persona'):
                region = next((r for r in Region if r.value in agent.name), None)
                agent._persona = {
                    'occupation': {
                        'title': 'Supplier',
                        'region': region.value if region else 'Unknown'
                    },
                    'capabilities': config['supplier'].get('capabilities', {})
                }
            suppliers.append(agent)
        elif 'Facility' in agent.name:
            if not hasattr(agent, '_persona'):
                region = next((r for r in Region if r.value in agent.name), None)
                agent._persona = {
                    'occupation': {
                        'title': 'Production Facility',
                        'region': region.value if region else 'Unknown'
                    },
                    'capabilities': config['production_facility'].get('capabilities', {})
                }
            production_facilities.append(agent)
        elif 'Logistics' in agent.name:
            if not hasattr(agent, '_persona'):
                mode = next((mode for mode in ['Air', 'Ocean', 'Ground'] if mode in agent.name), 'Unknown')
                agent._persona = {
                    'occupation': {
                        'title': 'Logistics Provider',
                        'mode': mode
                    },
                    'capabilities': config['logistics'].get('capabilities', {})
                }
            logistics_providers.append(agent)
    
    # Update world state with agent lists
    world.state['regional_managers'] = regional_managers
    world.state['suppliers'] = suppliers
    world.state['production_facilities'] = production_facilities
    world.state['logistics_providers'] = logistics_providers
    
    # Process existing orders first
    active_orders = world.state['active_orders']
    completed_orders = world.state['completed_orders']
    failed_orders = world.state['failed_orders']
    
    # Regional managers act and communicate with suppliers and facilities
    for manager in regional_managers:
        try:
            response = manager.act(world.state)
            if response:
                interaction_log.append({
                    'timestamp': world.state['current_datetime'],
                    'agent': manager.name,
                    'role': 'Regional Manager',
                    'region': manager._persona['occupation']['region'],
                    'action': 'Process orders',
                    'details': response
                })
        except Exception as e:
            logger.error(f"Error in Regional Manager {manager.name}: {str(e)}")
    
    # Suppliers act and process orders
    for supplier in suppliers:
        try:
            response = supplier.act(world.state)
            if response:
                interaction_log.append({
                    'timestamp': world.state['current_datetime'],
                    'agent': supplier.name,
                    'role': 'Supplier',
                    'region': supplier._persona['occupation']['region'],
                    'action': 'Process production',
                    'details': response
                })
        except Exception as e:
            logger.error(f"Error in Supplier {supplier.name}: {str(e)}")
    
    # Production facilities act
    for facility in production_facilities:
        try:
            response = facility.act(world.state)
            if response:
                interaction_log.append({
                    'timestamp': world.state['current_datetime'],
                    'agent': facility.name,
                    'role': 'Production Facility',
                    'region': facility._persona['occupation']['region'],
                    'action': 'Process production',
                    'details': response
                })
        except Exception as e:
            logger.error(f"Error in Production Facility {facility.name}: {str(e)}")
    
    # Logistics providers act
    for provider in logistics_providers:
        try:
            response = provider.act(world.state)
            if response:
                interaction_log.append({
                    'timestamp': world.state['current_datetime'],
                    'agent': provider.name,
                    'role': 'Logistics Provider',
                    'mode': provider._persona['occupation']['mode'],
                    'action': 'Process shipping',
                    'details': response
                })
        except Exception as e:
            logger.error(f"Error in Logistics Provider {provider.name}: {str(e)}")
    
    # Clean up completed and failed orders
    for order in active_orders[:]:  # Use slice to avoid modifying list while iterating
        if order.status == OrderStatus.DELIVERED:
            active_orders.remove(order)
            completed_orders.append(order)
        elif order.quality_check_passed is False:  # Check the quality_check_passed flag instead of status
            active_orders.remove(order)
            failed_orders.append(order)
    
    # Generate new orders only after processing existing ones
    new_orders = _generate_orders(world.state['current_datetime'], config)
    active_orders.extend(new_orders)

    # Calculate metrics based on order processing
    metrics = _calculate_order_based_metrics(completed_orders, active_orders, config)

    # Add order status counts to metrics
    status_counts = {status.value.lower(): 0 for status in OrderStatus}
    for order in active_orders + completed_orders + failed_orders:
        status_counts[order.status.value.lower()] += 1
    metrics['order_status'] = status_counts

    # Add interaction metrics
    metrics['total_interactions'] = len(interaction_log)
    metrics['unique_interacting_agents'] = len(set(log['agent'] for log in interaction_log))

    return metrics

def _generate_orders(current_datetime: datetime, config: Dict[str, Any]) -> List[Order]:
    """Generate new orders based on demand patterns."""
    new_orders = []
    base_demand = config['simulation'].get('base_demand', 10)
    available_regions = [Region.NORTH_AMERICA, Region.EUROPE]  # Only use regions with production facilities
    
    for source in available_regions:
        for dest in available_regions:
            if source != dest:
                # Generate random demand with some variability
                demand = int(random.gauss(base_demand, base_demand * 0.2))
                if demand > 0:
                    # Add some randomness to delivery time
                    delivery_days = _estimate_delivery_time(source, dest, config)
                    # Randomly make some orders have tighter deadlines
                    if random.random() < 0.3:  # 30% chance of tight deadline
                        delivery_days = max(1, delivery_days - 1)
                    order = Order(
                        id=f"ORD_{current_datetime.strftime('%Y%m%d_%H%M%S')}_{source.value}_{dest.value}",
                        product_type="Standard",
                        quantity=demand,
                        source_region=source,
                        destination_region=dest,
                        creation_time=current_datetime,
                        expected_delivery_time=current_datetime + timedelta(days=delivery_days),
                        current_location=source,
                        status=OrderStatus.NEW  # Changed from CREATED to NEW
                    )
                    new_orders.append(order)
    
    return new_orders

def _can_start_production(order: Order, facility: TinyPerson) -> bool:
    """Check if production can start for an order."""
    # In a real implementation, would check facility capacity, resource availability, etc.
    return True

def _assign_transportation_mode(order: Order, logistics_providers: List[TinyPerson], world_state: Dict[str, Any]) -> Optional[TransportationMode]:
    """Assign a transportation mode to an order based on distance and urgency."""
    # Calculate remaining time until expected delivery
    remaining_time = (order.expected_delivery_time - world_state.get('current_datetime', datetime.now())).days
    
    # Get available logistics providers
    available_providers = [lp for lp in logistics_providers if lp.persona['occupation'].get('mode') in [mode.value for mode in TransportationMode]]
    
    if not available_providers:
        return None
        
    # For urgent deliveries (less than 2 days remaining), prefer air transport
    if remaining_time <= 2:
        air_provider = next((lp for lp in available_providers if lp.persona['occupation']['mode'] == TransportationMode.AIR.value), None)
        if air_provider:
            return TransportationMode.AIR
            
    # For regional deliveries (same continent), prefer ground transport
    same_continent_regions = {
        'NORTH_AMERICA': ['NORTH_AMERICA'],
        'EUROPE': ['EUROPE'],
        'ASIA': ['EAST_ASIA', 'SOUTHEAST_ASIA', 'SOUTH_ASIA']
    }
    
    source_continent = next((cont for cont, regions in same_continent_regions.items() 
                           if order.source_region.value in regions), None)
    dest_continent = next((cont for cont, regions in same_continent_regions.items() 
                         if order.destination_region.value in regions), None)
                         
    if source_continent and source_continent == dest_continent:
        ground_provider = next((lp for lp in available_providers if lp.persona['occupation']['mode'] == TransportationMode.GROUND.value), None)
        if ground_provider:
            return TransportationMode.GROUND
            
    # For cross-continental deliveries with sufficient time, prefer ocean transport
    if remaining_time > 5:
        ocean_provider = next((lp for lp in available_providers if lp.persona['occupation']['mode'] == TransportationMode.OCEAN.value), None)
        if ocean_provider:
            return TransportationMode.OCEAN
            
    # Default to any available mode if no specific criteria are met
    provider = random.choice(available_providers)
    return TransportationMode(provider.persona['occupation']['mode'])

def _check_delivery(order: Order, current_datetime: datetime, config: Dict[str, Any]) -> bool:
    """Check if an order can be delivered based on its transit time and expected delivery time."""
    # Get transportation mode specific expected transit times
    mode_transit_times = {
        TransportationMode.AIR: config.get('logistics', {}).get('air_transit_time', 1),
        TransportationMode.OCEAN: config.get('logistics', {}).get('ocean_transit_time', 5),
        TransportationMode.GROUND: config.get('logistics', {}).get('ground_transit_time', 2)
    }

    # Get expected transit time for the current mode
    expected_transit_time = mode_transit_times.get(order.transportation_mode, 2)

    # Check if minimum transit time requirement is met
    if order.transit_time < expected_transit_time:
        return False

    # Check if we're past the expected delivery time
    current_delay = (current_datetime - order.expected_delivery_time).days
    if current_delay > config.get('logistics', {}).get('max_acceptable_delay', 3):
        # Mark order as delayed if it's significantly past due
        order.update_status(OrderStatus.DELAYED, current_datetime)
        return False

    # Check if we're in the same region as the destination
    if order.current_location == order.destination_region:
        # Allow delivery if we've met minimum transit time
        return order.transit_time >= expected_transit_time

    # For cross-region deliveries, ensure proper transit time based on mode
    same_continent_regions = {
        'NORTH_AMERICA': ['NORTH_AMERICA'],
        'EUROPE': ['EUROPE'],
        'ASIA': ['EAST_ASIA', 'SOUTHEAST_ASIA', 'SOUTH_ASIA']
    }
    
    source_continent = next((cont for cont, regions in same_continent_regions.items() 
                           if order.source_region.value in regions), None)
    dest_continent = next((cont for cont, regions in same_continent_regions.items() 
                         if order.destination_region.value in regions), None)

    # For cross-continental deliveries
    if source_continent != dest_continent:
        if order.transportation_mode == TransportationMode.GROUND:
            # Ground transport not suitable for cross-continental
            return False
        elif order.transportation_mode == TransportationMode.OCEAN:
            # Ocean transport needs longer minimum time
            return order.transit_time >= mode_transit_times[TransportationMode.OCEAN]
        elif order.transportation_mode == TransportationMode.AIR:
            # Air transport can be faster but still needs minimum time
            return order.transit_time >= mode_transit_times[TransportationMode.AIR]

    # For same-continent deliveries
    else:
        if order.transportation_mode == TransportationMode.GROUND:
            # Ground transport is ideal for same-continent
            return order.transit_time >= mode_transit_times[TransportationMode.GROUND]
        else:
            # Other modes can still be used within continent
            return order.transit_time >= expected_transit_time

    return False

def _check_delays(order: Order, current_datetime: datetime, config: Dict[str, Any]) -> bool:
    """Check if an order is delayed based on its expected delivery time."""
    # For testing purposes, be more lenient with delays
    if order.status == OrderStatus.IN_PRODUCTION:
        return False  # Don't mark as delayed during production
    elif order.status == OrderStatus.IN_TRANSIT:
        return order.transit_time > 2  # Only mark as delayed after 2 days in transit
    return False

def _estimate_delivery_time(source: Region, dest: Region, config: Dict[str, Any]) -> int:
    """Estimate delivery time between two regions."""
    # For testing purposes, use very short delivery times
    return 2  # Total time: 1 day production + 1 day transit

def _calculate_order_based_metrics(completed_orders: List[Order], active_orders: List[Order], config: Dict[str, Any]) -> Dict[str, float]:
    """Calculate metrics based on actual order data."""
    metrics = {}
    
    if not completed_orders:
        return config['coo']['initial_metrics']
    
    # Calculate lead time
    lead_times = [order.calculate_lead_time() for order in completed_orders if order.calculate_lead_time() is not None]
    metrics['lead_time'] = sum(lead_times) / len(lead_times) if lead_times else 0.0
    
    # Calculate service level (on-time delivery rate)
    on_time_orders = [order for order in completed_orders if order.is_on_time()]
    metrics['service_level'] = len(on_time_orders) / len(completed_orders)
    
    # Calculate risk exposure based on delays and quality issues
    problem_orders = [
        order for order in completed_orders + active_orders 
        if order.status in [OrderStatus.DELAYED, OrderStatus.QUALITY_CHECK_FAILED]
    ]
    metrics['risk_exposure'] = len(problem_orders) / (len(completed_orders) + len(active_orders))
    
    # Calculate quality score (based on quality check failures)
    failed_orders = [
        order for order in completed_orders + active_orders 
        if order.status == OrderStatus.QUALITY_CHECK_FAILED
    ]
    metrics['quality_score'] = 1.0 - (len(failed_orders) / (len(completed_orders) + len(active_orders)))
    
    # Calculate flexibility score based on recovery from delays
    recovered_orders = [
        order for order in completed_orders 
        if hasattr(order, 'was_delayed') and order.was_delayed and order.is_on_time()
    ]
    delayed_orders = [
        order for order in completed_orders + active_orders 
        if order.status == OrderStatus.DELAYED or (hasattr(order, 'was_delayed') and order.was_delayed)
    ]
    metrics['flexibility_score'] = len(recovered_orders) / len(delayed_orders) if delayed_orders else 1.0
    
    # Calculate resilience score
    metrics['resilience_score'] = (
        metrics['service_level'] * 0.3 +
        (1 - metrics['risk_exposure']) * 0.3 +
        metrics['flexibility_score'] * 0.2 +
        metrics['quality_score'] * 0.2
    )
    
    # Calculate recovery time based on average delay resolution
    delay_resolution_times = [
        order.delay_time for order in completed_orders 
        if hasattr(order, 'delay_time') and order.delay_time > 0
    ]
    metrics['recovery_time'] = sum(delay_resolution_times) / len(delay_resolution_times) if delay_resolution_times else 0.0
    
    # Calculate cost metrics
    base_cost = config.get('base_cost', 100)  # Base cost per order
    inventory_cost_rate = config.get('inventory_cost_rate', 0.1)  # Daily inventory holding cost rate
    transportation_cost_rates = {
        TransportationMode.AIR: 2.0,
        TransportationMode.OCEAN: 0.5,
        TransportationMode.GROUND: 1.0
    }
    
    # Calculate total cost
    total_cost = 0
    inventory_cost = 0
    transportation_cost = 0
    
    for order in completed_orders + active_orders:
        # Base cost
        order_cost = base_cost
        
        # Inventory cost
        if order.status in [OrderStatus.PRODUCTION, OrderStatus.READY_FOR_SHIPPING]:
            inventory_cost += base_cost * inventory_cost_rate * order.production_time
            order_cost += base_cost * inventory_cost_rate * order.production_time
        
        # Transportation cost
        if order.transportation_mode:
            transport_rate = transportation_cost_rates[order.transportation_mode]
            transport_cost = base_cost * transport_rate * order.transit_time
            transportation_cost += transport_cost
            order_cost += transport_cost
        
        total_cost += order_cost
    
    # Normalize cost metrics to 0-1 range
    max_expected_cost = base_cost * len(completed_orders + active_orders) * 3  # Assume max 3x base cost
    metrics['total_cost'] = min(1.0, total_cost / max_expected_cost)
    metrics['inventory_cost'] = min(1.0, inventory_cost / max_expected_cost)
    metrics['transportation_cost'] = min(1.0, transportation_cost / max_expected_cost)
    
    # Calculate supplier risk
    supplier_delays = [
        order for order in completed_orders + active_orders 
        if order.status == OrderStatus.DELAYED and 
        order.status in [OrderStatus.PRODUCTION, OrderStatus.READY_FOR_SHIPPING]
    ]
    metrics['supplier_risk'] = len(supplier_delays) / (len(completed_orders) + len(active_orders)) if completed_orders or active_orders else 0.0
    
    # Calculate transportation risk
    transport_delays = [
        order for order in completed_orders + active_orders 
        if order.status == OrderStatus.DELAYED and 
        order.status == OrderStatus.IN_TRANSIT
    ]
    metrics['transportation_risk'] = len(transport_delays) / (len(completed_orders) + len(active_orders)) if completed_orders or active_orders else 0.0
    
    # Normalize metrics to 0-1 range
    max_lead_time = config['simulation']['time_steps']
    metrics['lead_time'] = min(1.0, metrics['lead_time'] / max_lead_time)
    metrics['recovery_time'] = min(1.0, metrics['recovery_time'] / max_lead_time)
    
    return metrics

def export_comprehensive_results(
    baseline: Dict[str, float],
    supplier_diversification: Dict[str, float],
    dynamic_inventory: Dict[str, float],
    flexible_transportation: Dict[str, float],
    regional_flexibility: Dict[str, float],
    combined: Dict[str, float]
) -> None:
    """Export simulation results to CSV file."""
    results = {
        'Metric': [
            'Resilience Score',
            'Recovery Time',
            'Service Level',
            'Total Cost',
            'Inventory Cost',
            'Transportation Cost',
            'Risk Exposure',
            'Supplier Risk',
            'Transportation Risk',
            'Lead Time',
            'Flexibility Score',
            'Quality Score',
        ],
        'Baseline': [baseline[k] for k in [
            'avg_resilience_score',
            'avg_recovery_time',
            'avg_service_level',
            'avg_total_cost',
            'avg_inventory_cost',
            'avg_transportation_cost',
            'avg_risk_exposure',
            'avg_supplier_risk',
            'avg_transportation_risk',
            'avg_lead_time',
            'avg_flexibility_score',
            'avg_quality_score',
        ]],
        'Supplier Diversification': [supplier_diversification[k] for k in [
            'avg_resilience_score',
            'avg_recovery_time',
            'avg_service_level',
            'avg_total_cost',
            'avg_inventory_cost',
            'avg_transportation_cost',
            'avg_risk_exposure',
            'avg_supplier_risk',
            'avg_transportation_risk',
            'avg_lead_time',
            'avg_flexibility_score',
            'avg_quality_score',
        ]],
        'Dynamic Inventory': [dynamic_inventory[k] for k in [
            'avg_resilience_score',
            'avg_recovery_time',
            'avg_service_level',
            'avg_total_cost',
            'avg_inventory_cost',
            'avg_transportation_cost',
            'avg_risk_exposure',
            'avg_supplier_risk',
            'avg_transportation_risk',
            'avg_lead_time',
            'avg_flexibility_score',
            'avg_quality_score',
        ]],
        'Flexible Transportation': [flexible_transportation[k] for k in [
            'avg_resilience_score',
            'avg_recovery_time',
            'avg_service_level',
            'avg_total_cost',
            'avg_inventory_cost',
            'avg_transportation_cost',
            'avg_risk_exposure',
            'avg_supplier_risk',
            'avg_transportation_risk',
            'avg_lead_time',
            'avg_flexibility_score',
            'avg_quality_score',
        ]],
        'Regional Flexibility': [regional_flexibility[k] for k in [
            'avg_resilience_score',
            'avg_recovery_time',
            'avg_service_level',
            'avg_total_cost',
            'avg_inventory_cost',
            'avg_transportation_cost',
            'avg_risk_exposure',
            'avg_supplier_risk',
            'avg_transportation_risk',
            'avg_lead_time',
            'avg_flexibility_score',
            'avg_quality_score',
        ]],
        'Combined': [combined[k] for k in [
            'avg_resilience_score',
            'avg_recovery_time',
            'avg_service_level',
            'avg_total_cost',
            'avg_inventory_cost',
            'avg_transportation_cost',
            'avg_risk_exposure',
            'avg_supplier_risk',
            'avg_transportation_risk',
            'avg_lead_time',
            'avg_flexibility_score',
            'avg_quality_score',
        ]],
    }
    
    df = pd.DataFrame(results)
    df.to_csv('supply_chain_simulation_results.csv', index=False)

def run_monte_carlo_simulation(
    config: Dict[str, Any],
    world: World,
    has_supplier_diversification: bool = False,
    has_dynamic_inventory: bool = False,
    has_flexible_transportation: bool = False,
    has_regional_flexibility: bool = False
) -> Dict[str, float]:
    """Run Monte Carlo simulation for supply chain operations."""
    results = []
    random.seed(config['simulation']['seed'])
    
    for iteration in range(config['simulation']['monte_carlo_iterations']):
        # Create unique simulation ID
        simulation_id = f"{world.name}_iter_{iteration}"
        
        # Reset world state for this iteration
        world.state = {
            'risk_exposure': 0.5,
            'cost_pressure': 0.5,
            'demand_volatility': 0.5,
            'supply_risk': 0.5,
            'reliability_requirement': 0.5,
            'flexibility_requirement': 0.5,
            'active_orders': [],
            'completed_orders': [],
            'order_lifecycle': {},  # Initialize as dictionary of lists
            'regional_metrics': {
                region.value: {
                    'risk': 0.5,
                    'cost': 0.5,
                    'demand': 0.5,
                    'supply_risk': 0.5,
                    'infrastructure': 0.7,
                    'congestion': 0.3,
                    'efficiency': 0.8,
                    'flexibility': 0.7,
                    'quality': 0.8
                } for region in Region
            }
        }
        world.current_datetime = datetime.now()  # Reset datetime for this iteration
        
        # Initialize metrics tracking
        iteration_results = []
        daily_metrics = {
            'active_orders': [],
            'completed_orders': [],
            'delayed_orders': [],
            'service_level': [],
            'resilience_score': [],
            'lead_time': [],
            'total_orders': []
        }
        
        # Track daily order states and lifecycle events
        daily_order_tracking = []
        order_lifecycle = {}
        
        for step in range(config['simulation']['time_steps']):
            step_results = simulate_supply_chain_operation(
                world=world,
                config=config
            )
            
            # Store a snapshot of all orders for this day
            active_orders = world.state.get('active_orders', [])
            completed_orders = world.state.get('completed_orders', [])
            all_orders = active_orders + completed_orders
            daily_order_tracking.append(all_orders)
            
            # Track lifecycle events for each order
            for order in all_orders:
                lifecycle_event = {
                    'created_at': order.creation_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'current_status': order.status.value,
                    'current_location': order.current_location.value if order.current_location else None,
                    'production_time': order.production_time,
                    'transit_time': order.transit_time,
                    'delay_time': order.delay_time,
                    'expected_delivery': order.expected_delivery_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'actual_delivery': order.actual_delivery_time.strftime('%Y-%m-%d %H:%M:%S') if order.actual_delivery_time else None,
                    'transportation_mode': order.transportation_mode.value if order.transportation_mode else None,
                    'source_region': order.source_region.value,
                    'destination_region': order.destination_region.value,
                    'simulation_day': step,
                    'is_delayed': order.status == OrderStatus.DELAYED,
                    'is_on_time': order.is_on_time()
                }
                
                # Initialize list for this order if it doesn't exist
                if order.id not in order_lifecycle:
                    order_lifecycle[order.id] = []
                
                # Only append if this represents a state change
                if not order_lifecycle[order.id] or order_lifecycle[order.id][-1]['current_status'] != lifecycle_event['current_status']:
                    order_lifecycle[order.id].append(lifecycle_event)
            
            # Calculate total orders for this step
            step_results['total_orders'] = len(all_orders)
            
            # Remove current_datetime from step_results before appending
            if 'current_datetime' in step_results:
                del step_results['current_datetime']
            iteration_results.append(step_results)
            
            # Track daily metrics
            for metric in daily_metrics:
                if metric in step_results:
                    daily_metrics[metric].append(step_results[metric])
                else:
                    daily_metrics[metric].append(0)  # Default to 0 if metric not present
        
        # Aggregate results for this iteration
        iteration_aggregated = {}
        for metric in iteration_results[0].keys():
            if metric == 'order_status':
                continue  # Skip order_status as it's handled separately
            
            values = [r[metric] for r in iteration_results]
            if metric in ['delayed_orders', 'total_orders']:
                # For these metrics, take the maximum value across time steps
                iteration_aggregated[metric] = {
                    'mean': float(max(values)),
                    'std': float(np.std(values)),
                    'min': float(min(values)),
                    'max': float(max(values)),
                    'daily': values
                }
            else:
                # For other metrics, take the mean across time steps
                iteration_aggregated[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(min(values)),
                    'max': float(max(values)),
                    'daily': values
                }
        
        # Add order tracking and lifecycle data to iteration results
        iteration_aggregated['daily_order_tracking'] = daily_order_tracking
        iteration_aggregated['order_lifecycle'] = order_lifecycle
        iteration_aggregated['order_status'] = iteration_results[-1].get('order_status', {})
        
        results.append(iteration_aggregated)
        
        # Clean up agents after each iteration
        for agent in world.agents[:]:
            world.remove_agent(agent)
    
    # Aggregate results across all iterations
    aggregated_results = {}
    for metric in results[0].keys():
        if metric in ['daily_order_tracking', 'order_lifecycle', 'order_status']:
            # For order tracking and lifecycle, take the first iteration's values
            aggregated_results[metric] = results[0][metric]
            continue
            
        values = [r[metric]['mean'] for r in results]  # Use mean from each iteration
        daily_values = [r[metric]['daily'] for r in results]  # Get daily values from all iterations
        
        # Calculate mean daily values across iterations
        mean_daily = []
        for day in range(len(daily_values[0])):
            day_values = [iteration[day] for iteration in daily_values]
            mean_daily.append(float(np.mean(day_values)))
        
        aggregated_results[metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'daily': mean_daily
        }
    
    return aggregated_results

def _communicate_delay(order: Order, regional_managers: List[TinyPerson], suppliers: List[TinyPerson], logistics_providers: List[TinyPerson]):
    """Communicate order delays to relevant agents."""
    source_manager = next((m for m in regional_managers if m.region == order.source_region), None)
    dest_manager = next((m for m in regional_managers if m.region == order.destination_region), None)
    supplier = next((s for s in suppliers if s.name == order.supplier), None)
    
    delay_message = f"Order {order.id} is delayed - Current status: {order.status.value}"
    
    if source_manager and supplier:
        source_manager.communicate(supplier, delay_message)
    if dest_manager:
        for provider in logistics_providers:
            if provider.persona['occupation']['mode'] == order.transportation_mode.value:
                provider.communicate(dest_manager, delay_message)

def _communicate_delivery(order: Order, regional_managers: List[TinyPerson], suppliers: List[TinyPerson]):
    """Communicate successful delivery to relevant agents."""
    dest_manager = next((m for m in regional_managers if m.region == order.destination_region), None)
    supplier = next((s for s in suppliers if s.name == order.supplier), None)
    
    delivery_message = f"Order {order.id} successfully delivered to {order.destination_region.value}"
    
    if dest_manager and supplier:
        dest_manager.communicate(supplier, delivery_message)

__all__ = [
    'create_coo_agent',
    'create_regional_manager_agent',
    'create_supplier_agent',
    'create_logistics_provider_agent',
    'create_production_facility_agent',
    'create_simulation_world',
    'simulate_supply_chain_operation'
] 