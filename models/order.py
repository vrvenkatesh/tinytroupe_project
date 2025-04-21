from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from models.enums import Region, OrderStatus, TransportationMode

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