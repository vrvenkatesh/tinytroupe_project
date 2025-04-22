"""Transportation Agent for shipping products between regions."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from tinytroupe.agent import TinyPerson
from agents.base import BaseAgent
from models.order import Order, OrderStatus
from models.enums import Region

@dataclass
class TransportationAgent(TinyPerson, BaseAgent):
    """Agent representing a transportation service that can ship products between regions."""
    name: str
    config: Dict
    simulation_id: str
    speed: float = field(init=False)
    reliability: float = field(init=False)
    current_orders: Dict[str, Order] = field(default_factory=dict)
    completed_orders: List[str] = field(default_factory=list)
    shipping_times: Dict[str, float] = field(default_factory=dict)

    def __init__(self, name: str, config: Dict[str, Any], simulation_id: str):
        """Initialize the transportation agent."""
        super().__init__(name=name)
        self.config = config
        self.simulation_id = simulation_id
        transport_config = self.config['transportation']
        self.speed = transport_config['speed']
        self.reliability = transport_config['reliability']
        self.current_orders = {}
        self.completed_orders = []
        self.shipping_times = {}

    def process_order(self, order: Order, current_time: datetime) -> bool:
        """
        Process a new shipping order.
        Returns True if order was accepted, False otherwise.
        """
        # Verify order is in a valid state for shipping
        if order.status not in [OrderStatus.READY_FOR_SHIPPING]:
            return False

        # Calculate shipping time based on various factors
        shipping_time = self._calculate_shipping_time(order)
        
        # Update order and transportation state
        order.update_status(OrderStatus.IN_TRANSIT, current_time, self.name)
        self.current_orders[order.id] = order
        self.shipping_times[order.id] = shipping_time
        
        return True

    def _calculate_shipping_time(self, order: Order) -> float:
        """Calculate shipping time based on various factors."""
        # Base calculation components
        base_time = self._calculate_base_shipping_time(order.source_region, order.destination_region)
        speed_factor = 1 / self.speed  # Lower speed increases time
        size_factor = 1 + (order.quantity / 100) * 0.2  # Every 100 units adds 20% time
        
        # Reliability factor - higher reliability reduces variance
        reliability_factor = 1 + ((1 - self.reliability) * 0.3)  # Up to 30% increase for low reliability
        
        # Calculate final time
        shipping_time = base_time * speed_factor * size_factor * reliability_factor
        return max(1, round(shipping_time))  # Ensure minimum 1 time unit

    def _calculate_base_shipping_time(self, source: Region, destination: Region) -> float:
        """Calculate base shipping time between regions."""
        # Simple distance calculation for now
        # Could be enhanced with actual distances or shipping routes
        if source == destination:
            return 1
        return 2  # Base time between different regions

    def update(self, current_time: datetime) -> List[Order]:
        """
        Update shipping status and return completed orders.
        """
        completed_orders = []
        for order_id, order in list(self.current_orders.items()):
            self.shipping_times[order_id] -= 1
            if self.shipping_times[order_id] <= 0:
                order.update_status(OrderStatus.DELIVERED, current_time, self.name)
                order.current_location = order.destination_region
                completed_orders.append(order)
                self.completed_orders.append(order_id)
                del self.current_orders[order_id]
                del self.shipping_times[order_id]
        
        return completed_orders

    def get_reliability_score(self) -> float:
        """Return the transportation service's reliability score."""
        return self.reliability 