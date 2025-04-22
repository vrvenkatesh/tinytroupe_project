from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum

from models.enums import Region, OrderStatus, TransportationMode

class OrderPermission(Enum):
    """Permissions for order operations."""
    CREATE = "create"
    UPDATE_STATUS = "update_status"
    ASSIGN_PRODUCTION = "assign_production"
    UPDATE_LOCATION = "update_location"
    CANCEL = "cancel"
    QUALITY_CHECK = "quality_check"

ROLE_PERMISSIONS = {
    "COO": {
        OrderPermission.CREATE,
        OrderPermission.CANCEL,
        OrderPermission.UPDATE_STATUS
    },
    "RegionalManager": {
        OrderPermission.CREATE,
        OrderPermission.UPDATE_STATUS,
        OrderPermission.ASSIGN_PRODUCTION,
        OrderPermission.UPDATE_LOCATION,
        OrderPermission.CANCEL
    },
    "Supplier": {
        OrderPermission.UPDATE_STATUS,
        OrderPermission.QUALITY_CHECK,
        OrderPermission.UPDATE_LOCATION
    }
}

VALID_STATUS_TRANSITIONS = {
    OrderStatus.NEW: {OrderStatus.PRODUCTION, OrderStatus.CANCELLED},
    OrderStatus.PRODUCTION: {OrderStatus.READY_FOR_SHIPPING, OrderStatus.DELAYED, OrderStatus.QUALITY_CHECK_FAILED},
    OrderStatus.READY_FOR_SHIPPING: {OrderStatus.IN_TRANSIT, OrderStatus.DELAYED},
    OrderStatus.IN_TRANSIT: {OrderStatus.DELIVERED, OrderStatus.DELAYED},
    OrderStatus.DELAYED: {OrderStatus.PRODUCTION, OrderStatus.READY_FOR_SHIPPING, OrderStatus.IN_TRANSIT},
    OrderStatus.QUALITY_CHECK_FAILED: {OrderStatus.PRODUCTION, OrderStatus.CANCELLED},
    OrderStatus.DELIVERED: set(),  # Terminal state
    OrderStatus.CANCELLED: set()   # Terminal state
}

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
    status: OrderStatus = OrderStatus.NEW
    transportation_mode: Optional[TransportationMode] = None
    current_location: Optional[Region] = None
    production_time: int = 0  # Time spent in production
    transit_time: int = 0  # Time spent in transit
    delay_time: int = 0  # Total delay time
    cost: float = 0.0  # Total cost of the order
    production_facility: Optional[str] = None
    quality_check_passed: Optional[bool] = None  # Result of quality check
    _status_history: List[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize status history."""
        if self._status_history is None:
            self._status_history = [{
                'status': self.status,
                'timestamp': self.creation_time,
                'location': self.current_location
            }]

    def has_permission(self, agent_role: str, permission: OrderPermission) -> bool:
        """Check if an agent role has a specific permission."""
        if agent_role not in ROLE_PERMISSIONS:
            return False
        return permission in ROLE_PERMISSIONS[agent_role]

    def is_valid_transition(self, new_status: OrderStatus) -> bool:
        """Check if status transition is valid."""
        if self.status == new_status:
            return True
        return new_status in VALID_STATUS_TRANSITIONS.get(self.status, set())

    def update_status(self, new_status: OrderStatus, current_time: datetime, agent_role: str) -> bool:
        """Update order status with permission validation."""
        if not self.has_permission(agent_role, OrderPermission.UPDATE_STATUS):
            raise PermissionError(f"{agent_role} does not have permission to update order status")

        if not self.is_valid_transition(new_status):
            raise ValueError(f"Invalid status transition from {self.status} to {new_status}")

        # Special permission checks for specific transitions
        if new_status == OrderStatus.CANCELLED and not self.has_permission(agent_role, OrderPermission.CANCEL):
            raise PermissionError(f"{agent_role} does not have permission to cancel orders")
            
        if new_status == OrderStatus.QUALITY_CHECK_FAILED and not self.has_permission(agent_role, OrderPermission.QUALITY_CHECK):
            raise PermissionError(f"{agent_role} does not have permission to perform quality checks")

        # Update status and record in history
        old_status = self.status
        self.status = new_status
        self._status_history.append({
            'status': new_status,
            'timestamp': current_time,
            'location': self.current_location,
            'agent_role': agent_role
        })

        # Update timing metrics
        if new_status == OrderStatus.DELIVERED and self.actual_delivery_time is None:
            self.actual_delivery_time = current_time
            
        if new_status == OrderStatus.DELAYED:
            self.delay_time += 1

        return True

    def assign_production_facility(self, facility_id: str, agent_role: str) -> bool:
        """Assign order to a production facility."""
        if not self.has_permission(agent_role, OrderPermission.ASSIGN_PRODUCTION):
            raise PermissionError(f"{agent_role} does not have permission to assign production facilities")
            
        if self.status != OrderStatus.NEW:
            raise ValueError("Can only assign production facility to new orders")
            
        self.production_facility = facility_id
        return True

    def update_location(self, new_location: Region, agent_role: str) -> bool:
        """Update order's current location."""
        if not self.has_permission(agent_role, OrderPermission.UPDATE_LOCATION):
            raise PermissionError(f"{agent_role} does not have permission to update order location")
            
        self.current_location = new_location
        self._status_history[-1]['location'] = new_location
        return True

    def get_status_history(self) -> List[Dict[str, Any]]:
        """Get the complete status history of the order."""
        return self._status_history.copy()

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