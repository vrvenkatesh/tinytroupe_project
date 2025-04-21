"""Tests for simulation world functions."""

import pytest
from datetime import datetime

from simulation.world import (
    create_simulation_world,
    simulate_supply_chain_operation,
    _generate_orders,
    _check_delivery,
    _check_delays,
    _estimate_delivery_time,
    _calculate_order_based_metrics
)
from simulation.config import DEFAULT_CONFIG
from models.enums import Region, OrderStatus
from models.order import Order
from agents import COOAgent, RegionalManagerAgent, SupplierAgent, ExternalEventAgent

@pytest.fixture
def simulation_world():
    """Create a simulation world for testing."""
    return create_simulation_world(DEFAULT_CONFIG)

def test_create_simulation_world(simulation_world):
    """Test simulation world creation."""
    # Check that all necessary agents are created
    agent_types = [type(agent) for agent in simulation_world.agents]
    
    assert any(issubclass(t, COOAgent) for t in agent_types)
    assert any(issubclass(t, RegionalManagerAgent) for t in agent_types)
    assert any(issubclass(t, SupplierAgent) for t in agent_types)
    assert any(issubclass(t, ExternalEventAgent) for t in agent_types)
    
    # Check number of agents
    regional_managers = [a for a in simulation_world.agents if isinstance(a, RegionalManagerAgent)]
    suppliers = [a for a in simulation_world.agents if isinstance(a, SupplierAgent)]
    
    assert len(regional_managers) == len(Region)
    assert len(suppliers) == len(Region) * DEFAULT_CONFIG['simulation']['suppliers_per_region']

def test_simulate_supply_chain_operation(simulation_world):
    """Test single step of supply chain simulation."""
    metrics = simulate_supply_chain_operation(simulation_world, DEFAULT_CONFIG)
    
    # Check that metrics are calculated correctly
    assert 'completion_rate' in metrics
    assert 'on_time_delivery_rate' in metrics
    assert 'average_delay' in metrics
    
    assert 0 <= metrics['completion_rate'] <= 1
    assert 0 <= metrics['on_time_delivery_rate'] <= 1
    assert metrics['average_delay'] >= 0

def test_generate_orders():
    """Test order generation."""
    current_time = datetime.now()
    orders = _generate_orders(current_time, DEFAULT_CONFIG)
    
    assert len(orders) > 0
    for order in orders:
        assert isinstance(order, Order)
        assert order.source_region != order.destination_region
        assert order.status == OrderStatus.NEW
        assert order.creation_time == current_time
        assert order.expected_delivery_time > current_time

def test_check_delivery():
    """Test delivery check logic."""
    current_time = datetime.now()
    order = Order(
        id="test_order",
        product_type="Standard",
        quantity=10,
        source_region=Region.NORTH_AMERICA,
        destination_region=Region.EUROPE,
        creation_time=current_time,
        expected_delivery_time=current_time,
        status=OrderStatus.IN_TRANSIT
    )
    
    # Order should not be delivered yet
    assert not _check_delivery(order, current_time, DEFAULT_CONFIG)
    
    # Order should be delivered after expected time
    expected_time = _estimate_delivery_time(order.source_region, order.destination_region, DEFAULT_CONFIG)
    future_time = datetime.fromtimestamp(current_time.timestamp() + expected_time * 24 * 3600)
    assert _check_delivery(order, future_time, DEFAULT_CONFIG)

def test_check_delays():
    """Test delay check logic."""
    current_time = datetime.now()
    order = Order(
        id="test_order",
        product_type="Standard",
        quantity=10,
        source_region=Region.NORTH_AMERICA,
        destination_region=Region.EUROPE,
        creation_time=current_time,
        expected_delivery_time=current_time,
        status=OrderStatus.PRODUCTION
    )
    
    # Order should not be delayed initially
    assert not _check_delays(order, current_time, DEFAULT_CONFIG)
    
    # Order should be delayed after 1.5x expected time
    expected_time = _estimate_delivery_time(order.source_region, order.destination_region, DEFAULT_CONFIG)
    delayed_time = datetime.fromtimestamp(current_time.timestamp() + expected_time * 1.6 * 24 * 3600)
    assert _check_delays(order, delayed_time, DEFAULT_CONFIG)

def test_estimate_delivery_time():
    """Test delivery time estimation."""
    # Test standard route
    time_na_eu = _estimate_delivery_time(Region.NORTH_AMERICA, Region.EUROPE, DEFAULT_CONFIG)
    assert time_na_eu > 0
    
    # Test route with distance factor
    time_asia_na = _estimate_delivery_time(Region.EAST_ASIA, Region.NORTH_AMERICA, DEFAULT_CONFIG)
    assert time_asia_na > time_na_eu

def test_calculate_order_based_metrics():
    """Test metrics calculation."""
    current_time = datetime.now()
    
    # Create sample orders
    completed_orders = [
        Order(
            id=f"completed_{i}",
            product_type="Standard",
            quantity=10,
            source_region=Region.NORTH_AMERICA,
            destination_region=Region.EUROPE,
            creation_time=current_time,
            expected_delivery_time=current_time,
            status=OrderStatus.DELIVERED,
            actual_delivery_time=current_time
        )
        for i in range(3)
    ]
    
    active_orders = [
        Order(
            id=f"active_{i}",
            product_type="Standard",
            quantity=10,
            source_region=Region.NORTH_AMERICA,
            destination_region=Region.EUROPE,
            creation_time=current_time,
            expected_delivery_time=current_time,
            status=OrderStatus.PRODUCTION
        )
        for i in range(2)
    ]
    
    metrics = _calculate_order_based_metrics(completed_orders, active_orders, DEFAULT_CONFIG)
    
    assert metrics['completion_rate'] == 0.6  # 3 out of 5 orders completed
    assert 0 <= metrics['on_time_delivery_rate'] <= 1
    assert metrics['average_delay'] >= 0 