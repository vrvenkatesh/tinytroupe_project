"""Tests for the Transportation Agent."""

import pytest
from datetime import datetime, timedelta
from typing import Dict

from agents.transportation import TransportationAgent
from models.order import Order, OrderStatus
from models.enums import Region

@pytest.fixture
def config() -> Dict:
    """Create a test configuration."""
    return {
        'transportation': {
            'speed': 1.0,
            'reliability': 0.8,
        }
    }

@pytest.fixture
def transportation_agent(config: Dict) -> TransportationAgent:
    """Create a test transportation agent."""
    return TransportationAgent(
        name="test_transport",
        config=config,
        simulation_id="test_sim"
    )

@pytest.fixture
def test_order() -> Order:
    """Create a test order."""
    current_time = datetime.now()
    return Order(
        id="test_order_1",
        product_type="Standard",
        quantity=100,
        source_region=Region.NORTH_AMERICA,
        destination_region=Region.EUROPE,
        creation_time=current_time,
        expected_delivery_time=current_time + timedelta(days=5),
        production_time=2.0,
        transit_time=48.0,
        status=OrderStatus.READY_FOR_SHIPPING,
        current_location=Region.NORTH_AMERICA
    )

def test_initialization(transportation_agent: TransportationAgent, config: Dict):
    """Test transportation agent initialization."""
    assert transportation_agent.name == "test_transport"
    assert transportation_agent.config == config
    assert transportation_agent.simulation_id == "test_sim"
    assert transportation_agent.speed == config['transportation']['speed']
    assert transportation_agent.reliability == config['transportation']['reliability']
    assert transportation_agent.current_orders == {}
    assert transportation_agent.completed_orders == []
    assert transportation_agent.shipping_times == {}

def test_process_order(transportation_agent: TransportationAgent, test_order: Order):
    """Test processing a new order."""
    current_time = datetime.now()
    result = transportation_agent.process_order(test_order, current_time)
    
    assert result is True
    assert test_order.id in transportation_agent.current_orders
    assert test_order.id in transportation_agent.shipping_times
    assert test_order.status == OrderStatus.IN_TRANSIT
    assert test_order.status_update_time == current_time

def test_calculate_shipping_time(transportation_agent: TransportationAgent, test_order: Order):
    """Test shipping time calculation."""
    shipping_time = transportation_agent._calculate_shipping_time(test_order)
    
    # Base time for different regions is 2
    # Speed factor is 1/1.0 = 1
    # Size factor for 100 units is 1 + (100/100) * 0.2 = 1.2
    # Reliability factor for 0.8 reliability is 1 + (0.2 * 0.3) = 1.06
    # Expected: 2 * 1 * 1.2 * 1.06 ≈ 2.544 rounded to 3
    assert shipping_time == 3
    assert isinstance(shipping_time, (int, float))
    assert shipping_time >= 1  # Minimum shipping time

def test_calculate_base_shipping_time(transportation_agent: TransportationAgent):
    """Test base shipping time calculation."""
    # Same region
    time_same = transportation_agent._calculate_base_shipping_time(
        Region.NORTH_AMERICA, Region.NORTH_AMERICA
    )
    assert time_same == 1
    
    # Different regions
    time_diff = transportation_agent._calculate_base_shipping_time(
        Region.NORTH_AMERICA, Region.EUROPE
    )
    assert time_diff == 2

def test_update_completed_orders(transportation_agent: TransportationAgent, test_order: Order):
    """Test updating and completing orders."""
    current_time = datetime.now()
    
    # Process order
    transportation_agent.process_order(test_order, current_time)
    initial_shipping_time = transportation_agent.shipping_times[test_order.id]
    
    # Update until completion
    completed_orders = []
    for _ in range(initial_shipping_time):
        completed_orders = transportation_agent.update(current_time)
        if completed_orders:
            break
        current_time += timedelta(hours=1)
    
    assert len(completed_orders) == 1
    completed_order = completed_orders[0]
    assert completed_order.id == test_order.id
    assert completed_order.status == OrderStatus.DELIVERED
    assert completed_order.current_location == completed_order.destination_region
    assert test_order.id in transportation_agent.completed_orders
    assert test_order.id not in transportation_agent.current_orders
    assert test_order.id not in transportation_agent.shipping_times

def test_get_reliability_score(transportation_agent: TransportationAgent, config: Dict):
    """Test getting reliability score."""
    assert transportation_agent.get_reliability_score() == config['transportation']['reliability']

def test_multiple_orders(transportation_agent: TransportationAgent):
    """Test handling multiple orders simultaneously."""
    current_time = datetime.now()
    
    # Create multiple orders
    orders = [
        Order(
            id=f"test_order_{i}",
            product_type="Standard",
            quantity=50 + i * 25,  # Different quantities
            source_region=Region.NORTH_AMERICA,
            destination_region=Region.EUROPE,
            creation_time=current_time,
            expected_delivery_time=current_time + timedelta(days=5),
            production_time=2.0,
            transit_time=48.0,
            status=OrderStatus.READY_FOR_SHIPPING,
            current_location=Region.NORTH_AMERICA
        )
        for i in range(3)
    ]
    
    # Process all orders
    for order in orders:
        result = transportation_agent.process_order(order, current_time)
        assert result is True
    
    assert len(transportation_agent.current_orders) == 3
    assert len(transportation_agent.shipping_times) == 3
    
    # Verify different shipping times based on quantity
    shipping_times = [
        transportation_agent.shipping_times[order.id]
        for order in orders
    ]
    assert len(set(shipping_times)) > 1  # Different quantities should lead to different times 