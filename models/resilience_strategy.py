"""Models for supply chain resilience strategies."""

from datetime import timedelta
from dataclasses import dataclass
from models.enums import RiskLevel

@dataclass
class ResilienceStrategy:
    """Class representing a supply chain resilience strategy."""
    
    name: str
    description: str
    cost: float  # Cost to implement the strategy
    effectiveness: float  # 0.0 to 1.0
    implementation_time: timedelta
    risk_level: RiskLevel
    
    def __post_init__(self):
        """Validate strategy attributes."""
        if self.cost < 0:
            raise ValueError("Cost must be non-negative")
        if not 0 <= self.effectiveness <= 1:
            raise ValueError("Effectiveness must be between 0 and 1")
        if self.implementation_time.total_seconds() <= 0:
            raise ValueError("Implementation time must be positive") 