"""Tests for the Order model."""

import pytest
from datetime import datetime, timedelta

from models.enums import Region, OrderStatus, TransportationMode
from models.order import Order

@pytest.fixture
def sample_order():
    """Create a sample order for testing."""
    creation_time = datetime(2025, 4, 22, 13, 4, 29, 512058)
    return Order(
        id="test_order_1",
        product_type="Standard",
        quantity=10,
        source_region=Region.NORTH_AMERICA,
        destination_region=Region.EUROPE,
        creation_time=creation_time,
        expected_delivery_time=creation_time + timedelta(days=5)
    )

def test_order_initialization(sample_order):
    """Test order initialization."""
    assert sample_order.id == "test_order_1"
    assert sample_order.product_type == "Standard"
    assert sample_order.quantity == 10
    assert sample_order.source_region == Region.NORTH_AMERICA
    assert sample_order.destination_region == Region.EUROPE
    assert sample_order.status == OrderStatus.NEW
    # Order should start at source location
    assert sample_order.current_location == Region.NORTH_AMERICA
    assert sample_order._status_history[0]['status'] == OrderStatus.NEW

def test_order_status_update(sample_order):
    """Test order status updates follow valid transitions."""
    current_time = datetime.now()
    
    # Follow the valid status transition path
    sample_order.update_status(OrderStatus.PRODUCTION, current_time)
    assert sample_order.status == OrderStatus.PRODUCTION
    
    sample_order.update_status(OrderStatus.READY_FOR_SHIPPING, current_time)
    assert sample_order.status == OrderStatus.READY_FOR_SHIPPING
    
    sample_order.update_status(OrderStatus.IN_TRANSIT, current_time)
    assert sample_order.status == OrderStatus.IN_TRANSIT
    
    sample_order.update_status(OrderStatus.DELIVERED, current_time)
    assert sample_order.status == OrderStatus.DELIVERED
    assert len(sample_order._status_history) == 5  # Initial + 4 updates

def test_order_lead_time_calculation(sample_order):
    """Test lead time calculation for delivered orders."""
    current_time = datetime.now()
    
    # Move through status transitions
    sample_order.update_status(OrderStatus.PRODUCTION, current_time)
    sample_order.update_status(OrderStatus.READY_FOR_SHIPPING, current_time)
    sample_order.update_status(OrderStatus.IN_TRANSIT, current_time)
    sample_order.update_status(OrderStatus.DELIVERED, current_time)
    
    lead_time = sample_order.calculate_lead_time()
    assert lead_time is not None
    assert isinstance(lead_time, int)

def test_order_on_time_delivery(sample_order):
    """Test on-time delivery check."""
    # Test early delivery
    early_time = sample_order.expected_delivery_time - timedelta(days=1)
    sample_order.update_status(OrderStatus.PRODUCTION, early_time)
    sample_order.update_status(OrderStatus.READY_FOR_SHIPPING, early_time)
    sample_order.update_status(OrderStatus.IN_TRANSIT, early_time)
    sample_order.update_status(OrderStatus.DELIVERED, early_time)
    assert sample_order.is_on_time() is True
    
    # Create a new order for late delivery test
    creation_time = datetime(2025, 4, 22, 13, 4, 29, 512058)
    expected_time = creation_time + timedelta(days=2)  # April 24, 2025
    late_time = expected_time + timedelta(days=1)  # April 25, 2025 (definitely late)
    
    late_order = Order(
        id="test_order_2",
        product_type="Standard",
        quantity=10,
        source_region=Region.NORTH_AMERICA,
        destination_region=Region.EUROPE,
        creation_time=creation_time,
        expected_delivery_time=expected_time
    )
    
    # Test late delivery
    late_order.update_status(OrderStatus.PRODUCTION, late_time)
    late_order.update_status(OrderStatus.READY_FOR_SHIPPING, late_time)
    late_order.update_status(OrderStatus.IN_TRANSIT, late_time)
    late_order.update_status(OrderStatus.DELIVERED, late_time)
    assert late_order.is_on_time() is False 