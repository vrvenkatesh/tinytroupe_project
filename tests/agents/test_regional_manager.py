"""Tests for the Regional Manager Agent."""

import pytest
from datetime import datetime, timedelta
from typing import Dict

from agents.regional_manager import RegionalManagerAgent
from models.order import Order, OrderStatus
from models.enums import Region

@pytest.fixture
def test_config() -> Dict:
    """Create a test configuration."""
    return {
        'regional_manager': {
            'order_batch_size': 5,
            'order_processing_interval': 1,
            'regional_demand_weights': {
                'North America': 0.3,
                'Europe': 0.3,
                'East Asia': 0.2,
                'Southeast Asia': 0.2
            },
            'regional_production_costs': {
                'North America': 100,
                'Europe': 120,
                'East Asia': 80,
                'Southeast Asia': 90
            }
        },
        'local_expertise': 0.8,
        'cost_sensitivity': 0.7,
        'adaptability': 0.9,
        'dynamic_enabled': True
    }

@pytest.fixture
def manager(test_config: Dict) -> RegionalManagerAgent:
    """Create a test regional manager."""
    return RegionalManagerAgent(
        name="test_manager",
        config=test_config,
        simulation_id="test_sim",
        region=Region.NORTH_AMERICA
    )

@pytest.fixture
def test_order():
    """Create a test order."""
    return Order(
        id="TEST_ORDER_001",
        product_type="Standard",
        quantity=10,
        source_region=Region.NORTH_AMERICA,
        destination_region=Region.EUROPE,
        creation_time=datetime.now(),
        expected_delivery_time=datetime.now() + timedelta(days=5),
        production_time=2.0,
        transit_time=1.0,
        status=OrderStatus.NEW,
        current_location=Region.NORTH_AMERICA
    )

def test_manager_initialization(manager: RegionalManagerAgent, test_config: Dict):
    """Test regional manager initialization."""
    assert manager.name == "test_manager"
    assert manager.config == test_config
    assert manager.simulation_id == "test_sim"
    assert manager.region == Region.NORTH_AMERICA
    assert manager.order_batch_size == test_config['regional_manager']['order_batch_size']
    assert manager.order_processing_interval == test_config['regional_manager']['order_processing_interval']
    assert manager.regional_demand_weights == test_config['regional_manager']['regional_demand_weights']
    assert manager.regional_production_costs == test_config['regional_manager']['regional_production_costs']

def test_receive_order(manager: RegionalManagerAgent, test_order: Order):
    """Test receiving an order."""
    manager.receive_order(test_order)
    assert len(manager.pending_orders) == 1
    assert manager.pending_orders[0].id == test_order.id

def test_process_orders(manager: RegionalManagerAgent, test_order: Order):
    """Test processing orders."""
    manager.receive_order(test_order)
    processed = manager.process_orders(datetime.now())
    
    assert len(processed) == 1
    assert processed[0].status == OrderStatus.PRODUCTION
    assert len(manager.pending_orders) == 0
    assert len(manager.processed_orders) == 1

def test_batch_processing(manager: RegionalManagerAgent):
    """Test batch processing of orders."""
    current_time = datetime.now()
    batch_size = manager.order_batch_size
    
    # Create more orders than batch size
    orders = [
        Order(
            id=f"TEST_ORDER_{i}",
            product_type="Standard",
            quantity=10,
            source_region=Region.NORTH_AMERICA,
            destination_region=Region.EUROPE,
            creation_time=current_time,
            expected_delivery_time=current_time + timedelta(days=5),
            status=OrderStatus.NEW,
            current_location=Region.NORTH_AMERICA
        )
        for i in range(batch_size + 2)
    ]
    
    for order in orders:
        manager.receive_order(order)
    
    processed = manager.process_orders(current_time)
    assert len(processed) == batch_size
    assert len(manager.pending_orders) == 2

def test_order_prioritization(manager: RegionalManagerAgent):
    """Test that orders are prioritized correctly."""
    # Create orders with different priorities (based on creation time)
    current_time = datetime.now()
    
    # Later order
    order1 = Order(
        id="TEST_ORDER_001",
        product_type="Standard",
        quantity=10,
        source_region=Region.NORTH_AMERICA,
        destination_region=Region.EUROPE,
        creation_time=current_time + timedelta(hours=1),
        expected_delivery_time=current_time + timedelta(days=5),
        status=OrderStatus.NEW,
        current_location=Region.NORTH_AMERICA
    )
    
    # Earlier order
    order2 = Order(
        id="TEST_ORDER_002",
        product_type="Standard",
        quantity=10,
        source_region=Region.NORTH_AMERICA,
        destination_region=Region.EUROPE,
        creation_time=current_time,
        expected_delivery_time=current_time + timedelta(days=5),
        status=OrderStatus.NEW,
        current_location=Region.NORTH_AMERICA
    )
    
    manager.receive_order(order1)
    manager.receive_order(order2)
    
    processed = manager.process_orders(current_time)
    assert processed[0].id == order2.id  # Earlier order should be processed first

def test_regional_cost_calculation(manager: RegionalManagerAgent):
    """Test regional cost calculation."""
    cost = manager.get_production_cost(Region.NORTH_AMERICA)
    assert cost == manager.regional_production_costs['North America']
    
    cost = manager.get_production_cost(Region.EUROPE)
    assert cost == manager.regional_production_costs['Europe']

def test_demand_weight_calculation(manager: RegionalManagerAgent):
    """Test demand weight calculation."""
    weight = manager.get_demand_weight(Region.NORTH_AMERICA)
    assert weight == manager.regional_demand_weights['North America']
    
    weight = manager.get_demand_weight(Region.EUROPE)
    assert weight == manager.regional_demand_weights['Europe']

def test_order_routing_decision(manager: RegionalManagerAgent, test_order: Order):
    """Test order routing decision."""
    routing_decision = manager.make_routing_decision(test_order)
    assert isinstance(routing_decision, Region)
    
    # East Asia should be chosen as it has lowest cost
    assert routing_decision == Region.EAST_ASIA 