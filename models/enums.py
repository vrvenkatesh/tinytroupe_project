from enum import Enum

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