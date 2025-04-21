"""Models for supply chain disruptions."""

from datetime import datetime, timedelta
from dataclasses import dataclass, field
import uuid
from models.enums import DisruptionType, Region

@dataclass
class Disruption:
    """Class representing a supply chain disruption event."""
    
    type: DisruptionType
    region: Region
    severity: float  # 0.0 to 1.0
    start_time: datetime
    expected_duration: timedelta
    affected_capacity: float  # 0.0 to 1.0
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __post_init__(self):
        """Validate disruption attributes."""
        if not 0 <= self.severity <= 1:
            raise ValueError("Severity must be between 0 and 1")
        if not 0 <= self.affected_capacity <= 1:
            raise ValueError("Affected capacity must be between 0 and 1")
        if self.expected_duration.total_seconds() <= 0:
            raise ValueError("Expected duration must be positive") 