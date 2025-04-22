from dataclasses import dataclass, field
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
    status: OrderStatus = OrderStatus.NEW
    current_location: Region = None
    production_time: float = 0.0
    transit_time: float = 0.0
    delay_time: float = 0.0
    actual_delivery_time: datetime = None
    transportation_mode: TransportationMode = None
    production_facility: str = None
    supplier: str = None
    current_handler: str = None  # Name of the agent currently handling the order
    status_history: List[Dict[str, Any]] = field(default_factory=list)  # Remove underscore
    status_update_time: datetime = None  # Track when the status was last updated
    
    def __post_init__(self):
        """Initialize after dataclass creation."""
        if self.current_location is None:
            self.current_location = self.source_region
        if self.status_history is None:  # Remove underscore
            self.status_history = []
        if self.status_update_time is None:
            self.status_update_time = self.creation_time
        # Add initial status to history
        self.status_history.append({  # Remove underscore
            'timestamp': self.creation_time,
            'status': self.status,
            'location': self.current_location,
            'handler': self.current_handler
        })

    def update_status(self, new_status: OrderStatus, timestamp: datetime, handler_name: str = None) -> bool:
        """Update order status with validation.
        
        Args:
            new_status: The new status to set
            timestamp: When the status change occurred
            handler_name: Name of the agent handling this status change
        """
        if new_status not in VALID_STATUS_TRANSITIONS.get(self.status, set()):
            raise ValueError(f"Invalid status transition from {self.status} to {new_status}")
        
        self.status = new_status
        self.status_update_time = timestamp  # Update the status update time
        if handler_name:
            self.current_handler = handler_name
        
        # Record in status history
        self.status_history.append({  # Remove underscore
            'timestamp': timestamp,
            'status': new_status,
            'location': self.current_location,
            'handler': handler_name
        })
        
        # Update delivery time if delivered
        if new_status == OrderStatus.DELIVERED:
            self.actual_delivery_time = timestamp
        
        return True

    def assign_supplier(self, supplier_id: str, agent_role: str, handler_name: str = None) -> bool:
        """Assign order to a supplier."""
        if not self.has_permission(agent_role, OrderPermission.ASSIGN_PRODUCTION):
            raise PermissionError(f"{agent_role} does not have permission to assign suppliers")
            
        if self.status != OrderStatus.NEW:
            raise ValueError("Can only assign supplier to new orders")
            
        self.supplier = supplier_id
        if handler_name:
            self.current_handler = handler_name
        return True

    def assign_production_facility(self, facility_id: str, agent_role: str, handler_name: str = None) -> bool:
        """Assign order to a production facility."""
        if not self.has_permission(agent_role, OrderPermission.ASSIGN_PRODUCTION):
            raise PermissionError(f"{agent_role} does not have permission to assign production facilities")
            
        if self.status != OrderStatus.NEW:
            raise ValueError("Can only assign production facility to new orders")
            
        self.production_facility = facility_id
        if handler_name:
            self.current_handler = handler_name
        return True

    def update_location(self, new_location: Region, agent_role: str, handler_name: str = None) -> bool:
        """Update order's current location."""
        if not self.has_permission(agent_role, OrderPermission.UPDATE_LOCATION):
            raise PermissionError(f"{agent_role} does not have permission to update order location")
            
        self.current_location = new_location
        if handler_name:
            self.current_handler = handler_name
        self.status_history[-1]['location'] = new_location
        return True

    def has_permission(self, agent_role: str, permission: OrderPermission) -> bool:
        """Check if an agent has permission for an operation."""
        if agent_role not in ROLE_PERMISSIONS:
            return False
        return permission in ROLE_PERMISSIONS[agent_role]

    def get_status_history(self) -> List[Dict[str, Any]]:
        """Get the complete status history of the order."""
        return self.status_history.copy()  # Remove underscore

    def calculate_lead_time(self) -> Optional[int]:
        """Calculate total lead time if order is delivered."""
        if self.status == OrderStatus.DELIVERED:
            return (datetime.now() - self.status_history[0]['timestamp']).days
        return None
    
    def is_on_time(self) -> Optional[bool]:
        """Check if order was delivered on time."""
        if self.status == OrderStatus.DELIVERED and self.actual_delivery_time:
            return self.actual_delivery_time <= self.expected_delivery_time
        return None 