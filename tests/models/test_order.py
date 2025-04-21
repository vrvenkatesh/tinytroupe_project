"""Tests for the Order model."""

import pytest
from datetime import datetime, timedelta

from models.enums import Region, OrderStatus, TransportationMode
from models.order import Order

@pytest.fixture
def sample_order():
    """Create a sample order for testing."""
    current_time = datetime.now()
    return Order(
        id="test_order_1",
        product_type="Standard",
        quantity=10,
        source_region=Region.NORTH_AMERICA,
        destination_region=Region.EUROPE,
        creation_time=current_time,
        expected_delivery_time=current_time + timedelta(days=5)
    )

def test_order_initialization(sample_order):
    """Test that an order is initialized with correct default values."""
    assert sample_order.id == "test_order_1"
    assert sample_order.product_type == "Standard"
    assert sample_order.quantity == 10
    assert sample_order.source_region == Region.NORTH_AMERICA
    assert sample_order.destination_region == Region.EUROPE
    assert sample_order.status == OrderStatus.NEW
    assert sample_order.transportation_mode is None
    assert sample_order.current_location is None
    assert sample_order.production_time == 0
    assert sample_order.transit_time == 0
    assert sample_order.delay_time == 0
    assert sample_order.cost == 0.0
    assert sample_order.production_facility is None
    assert sample_order.quality_check_passed is None

def test_order_status_update(sample_order):
    """Test order status updates and timing metrics."""
    current_time = datetime.now()
    
    # Test delivery status update
    sample_order.update_status(OrderStatus.DELIVERED, current_time)
    assert sample_order.status == OrderStatus.DELIVERED
    assert sample_order.actual_delivery_time == current_time
    
    # Test delay status update
    sample_order.update_status(OrderStatus.DELAYED, current_time)
    assert sample_order.status == OrderStatus.DELAYED
    assert sample_order.delay_time == 1

def test_order_lead_time_calculation(sample_order):
    """Test lead time calculation."""
    current_time = datetime.now()
    
    # Before delivery
    assert sample_order.calculate_lead_time() is None
    
    # After delivery
    sample_order.update_status(OrderStatus.DELIVERED, current_time)
    lead_time = sample_order.calculate_lead_time()
    assert isinstance(lead_time, int)
    assert lead_time >= 0

def test_order_on_time_delivery(sample_order):
    """Test on-time delivery check."""
    # Before delivery
    assert sample_order.is_on_time() is None
    
    # On-time delivery
    early_time = sample_order.expected_delivery_time - timedelta(days=1)
    sample_order.update_status(OrderStatus.DELIVERED, early_time)
    assert sample_order.is_on_time() is True
    
    # Create a new order for late delivery test
    current_time = datetime.now()
    late_order = Order(
        id="test_order_2",
        product_type="Standard",
        quantity=10,
        source_region=Region.NORTH_AMERICA,
        destination_region=Region.EUROPE,
        creation_time=current_time,
        expected_delivery_time=current_time + timedelta(days=5)
    )
    
    # Late delivery
    late_time = late_order.expected_delivery_time + timedelta(days=1)
    late_order.update_status(OrderStatus.DELIVERED, late_time)
    assert late_order.is_on_time() is False 