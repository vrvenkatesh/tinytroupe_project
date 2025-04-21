"""Enums for supply chain models."""

from enum import Enum, auto

class Region(Enum):
    """Geographic regions for supply chain operations."""
    NORTH_AMERICA = "North America"
    EUROPE = "Europe"
    EAST_ASIA = "East Asia"
    SOUTHEAST_ASIA = "Southeast Asia"
    SOUTH_ASIA = "South Asia"

class TransportationMode(Enum):
    OCEAN = "Ocean"
    AIR = "Air"
    GROUND = "Ground"

class DisruptionType(Enum):
    """Types of supply chain disruptions."""
    NATURAL_DISASTER = auto()
    SUPPLIER_BANKRUPTCY = auto()
    POLITICAL_UNREST = auto()
    LABOR_STRIKE = auto()
    CYBER_ATTACK = auto()
    TRANSPORTATION_FAILURE = auto()
    QUALITY_ISSUE = auto()
    DEMAND_SHOCK = auto()

class RiskLevel(Enum):
    """Risk levels for resilience strategies."""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()

class OrderStatus(Enum):
    """Status of supply chain orders."""
    NEW = "NEW"                      # Initial state when order is created
    PRODUCTION = "PRODUCTION"        # Order assigned to supplier and in production
    READY_FOR_SHIPPING = "READY"     # Production complete, ready to be shipped
    IN_TRANSIT = "IN_TRANSIT"        # Order is being shipped
    DELIVERED = "DELIVERED"          # Order has reached its destination
    CANCELLED = "CANCELLED"          # Order was cancelled
    DELAYED = "DELAYED"              # Order is delayed
    QUALITY_CHECK_FAILED = "FAILED"  # Order failed quality check 