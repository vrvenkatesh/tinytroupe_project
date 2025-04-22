from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime

from models.enums import Region, OrderStatus, TransportationMode
from models.order import Order

@dataclass
class Agent:
    """Base class for all agents in the supply chain simulation."""
    name: str
    region: Optional[Region] = None
    interactions: List[Dict[str, Any]] = None
    performance_metrics: Dict[str, float] = None
    agent_type: str = field(init=False)  # Set by subclasses
    
    def __post_init__(self):
        """Initialize after dataclass creation."""
        if self.interactions is None:
            self.interactions = []
        if self.performance_metrics is None:
            self.performance_metrics = {
                'orders_handled': 0,
                'success_rate': 1.0,
                'efficiency': 0.8
            }
    
    def record_interaction(self, interaction_type: str, timestamp: datetime, **kwargs) -> None:
        """Record an interaction in the agent's history."""
        interaction = {
            'id': str(uuid.uuid4()),
            'agent_id': self.name,
            'agent_type': self.agent_type,
            'interaction_type': interaction_type,
            'timestamp': timestamp,
            **kwargs
        }
        self.interactions.append(interaction)
        
        # Update performance metrics
        self.performance_metrics['orders_handled'] += 1 if kwargs.get('order_id') else 0
        self.performance_metrics['success_rate'] = (
            sum(1 for i in self.interactions if i.get('success', True)) / 
            len(self.interactions)
        )
    
    def get_interactions(self, start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None,
                        interaction_type: Optional[str] = None,
                        order_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get filtered interactions history.
        
        Args:
            start_time: Filter interactions after this time
            end_time: Filter interactions before this time
            interaction_type: Filter by interaction type
            order_id: Filter by order ID
            
        Returns:
            List of matching interactions
        """
        filtered = self.interactions
        
        if start_time:
            filtered = [i for i in filtered if i['timestamp'] >= start_time]
        if end_time:
            filtered = [i for i in filtered if i['timestamp'] <= end_time]
        if interaction_type:
            filtered = [i for i in filtered if i['interaction_type'] == interaction_type]
        if order_id:
            filtered = [i for i in filtered if i['order_id'] == order_id]
            
        return filtered
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get the agent's current performance metrics."""
        return self.performance_metrics.copy()
        
    def handle_order(self, order: Order, world_state: Dict[str, Any]) -> bool:
        """Handle an order based on agent's role and capabilities."""
        raise NotImplementedError("Subclasses must implement handle_order")

@dataclass
class COOAgent(Agent):
    """Chief Operating Officer agent responsible for global supply chain decisions."""
    config: Dict[str, Any] = field(default_factory=dict)
    simulation_id: str = None
    
    def __post_init__(self):
        self.agent_type = "COO"
        super().__post_init__()
        
    def handle_order(self, order: Order, world_state: Dict[str, Any]) -> bool:
        """Handle strategic decisions for order management."""
        # Record the interaction
        self.record_interaction(
            interaction_type="STRATEGIC_DECISION",
            timestamp=datetime.now(),
            order_id=order.id,
            status=order.status,
            message=f"COO reviewing order {order.id}"
        )
        return True

@dataclass
class RegionalManagerAgent(Agent):
    """Regional manager responsible for operations in a specific region."""
    config: Dict[str, Any] = field(default_factory=dict)
    simulation_id: str = None
    
    def __post_init__(self):
        if self.region is None:
            raise ValueError("RegionalManagerAgent requires a region")
        self.agent_type = "RegionalManager"
        super().__post_init__()
        
    def handle_order(self, order: Order, world_state: Dict[str, Any]) -> bool:
        """Handle regional order management."""
        # Record the interaction
        self.record_interaction(
            interaction_type="REGIONAL_MANAGEMENT",
            timestamp=datetime.now(),
            order_id=order.id,
            status=order.status,
            message=f"Managing order {order.id} in region {self.region.value}"
        )
        return True
        
    def manage_region(self, world_state: Dict[str, Any]) -> None:
        """Manage all aspects of regional operations."""
        pass

@dataclass
class SupplierAgent(Agent):
    """Supplier agent responsible for production and delivery."""
    config: Dict[str, Any] = field(default_factory=dict)
    simulation_id: str = None
    supplier_type: str = "tier1"
    
    def __post_init__(self):
        if self.region is None:
            raise ValueError("SupplierAgent requires a region")
        self.agent_type = "Supplier"
        super().__post_init__()
        
    def handle_order(self, order: Order, world_state: Dict[str, Any]) -> bool:
        """Handle order production and preparation for shipping."""
        # Record the interaction
        self.record_interaction(
            interaction_type="PRODUCTION",
            timestamp=datetime.now(),
            order_id=order.id,
            status=order.status,
            message=f"Processing order {order.id} for production"
        )
        return True 