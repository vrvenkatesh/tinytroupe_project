"""Production Facility Agent for manufacturing products."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import math

from tinytroupe.agent import TinyPerson
from agents.base import BaseAgent
from models.order import Order, OrderStatus
from models.enums import Region

@dataclass
class ProductionFacilityAgent(TinyPerson, BaseAgent):
    """Agent representing a production facility that can manufacture products."""
    name: str
    config: Dict
    simulation_id: str
    region: Region
    capacity: int = field(init=False)
    efficiency: float = field(init=False)
    quality_control: float = field(init=False)
    flexibility: float = field(init=False)
    base_production_time: float = field(init=False)
    current_load: int = field(default=0)
    current_orders: Dict[str, Order] = field(default_factory=dict)
    completed_orders: List[str] = field(default_factory=list)
    production_times: Dict[str, float] = field(default_factory=dict)

    def __init__(self, name: str, config: Dict[str, Any], simulation_id: str, region: Region):
        """Initialize the production facility agent."""
        super().__init__(name=name)
        self.config = config
        self.region = region
        self.simulation_id = simulation_id
        facility_config = self.config['production_facility']
        self.capacity = facility_config['capacity'][self.region.value]
        self.efficiency = facility_config['efficiency']
        self.quality_control = facility_config['quality_control']
        self.flexibility = facility_config['flexibility']
        self.base_production_time = facility_config['base_production_time']
        self.current_load = 0
        self.current_orders = {}
        self.completed_orders = []
        self.production_times = {}

    def process_order(self, order: Order, current_time: datetime) -> bool:
        """
        Process a new order if capacity allows.
        Returns True if order was accepted, False otherwise.
        """
        if order.quantity > self.get_capacity():
            return False

        # Calculate production time based on various factors
        production_time = self._calculate_production_time(order)
        
        # Update order and facility state
        order.update_status(OrderStatus.PRODUCTION, current_time, self.name)
        order.current_location = self.region
        self.current_orders[order.id] = order
        self.current_load += order.quantity
        self.production_times[order.id] = production_time
        
        return True

    def _calculate_production_time(self, order: Order) -> float:
        """Calculate production time based on various factors."""
        # Base calculation components
        base_time = self.base_production_time
        efficiency_factor = 1 / self.efficiency  # Lower efficiency increases time
        size_factor = math.ceil(order.quantity / 50)  # Every 50 units adds a time unit
        
        # Quality factor - higher quality control increases time
        quality_factor = 1 + (self.quality_control * 0.2)  # Up to 20% increase for quality
        
        # Flexibility bonus - higher flexibility reduces time
        flexibility_bonus = 1 - (self.flexibility * 0.3)  # Up to 30% reduction for flexibility
        
        # Calculate final time
        production_time = math.ceil(base_time * efficiency_factor * size_factor * quality_factor * flexibility_bonus)
        return max(1, production_time)  # Ensure minimum 1 time unit

    def update(self, current_time: datetime) -> List[Order]:
        """
        Update production status and return completed orders.
        """
        completed_orders = []
        for order_id, order in list(self.current_orders.items()):
            self.production_times[order_id] -= 1
            if self.production_times[order_id] <= 0:
                order.update_status(OrderStatus.READY_FOR_SHIPPING, current_time, self.name)
                completed_orders.append(order)
                self.completed_orders.append(order_id)
                self.current_load -= order.quantity
                del self.current_orders[order_id]
                del self.production_times[order_id]
        
        return completed_orders

    def get_capacity(self) -> int:
        """Return available production capacity."""
        return self.capacity - self.current_load 

    def get_quality_score(self) -> float:
        """Return the facility's quality score based on configuration."""
        return self.quality_control

    def get_flexibility_score(self) -> float:
        """Return the facility's flexibility score based on configuration."""
        return self.flexibility 