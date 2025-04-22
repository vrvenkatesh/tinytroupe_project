"""Tests for the Production Facility Agent."""

import pytest
from datetime import datetime, timedelta
from typing import Dict

from agents.production_facility import ProductionFacilityAgent
from models.order import Order, OrderStatus
from models.enums import Region

@pytest.fixture
def test_config() -> Dict:
    """Create a test configuration."""
    return {
        'production_facility': {
            'capacity': {
                'North America': 200,
                'Europe': 150,
                'East Asia': 300,
                'Southeast Asia': 250
            },
            'efficiency': 0.8,
            'quality_control': 0.9,
            'flexibility': 0.7,
            'base_production_time': 2.0
        }
    }

@pytest.fixture
def facility(test_config: Dict) -> ProductionFacilityAgent:
    """Create a test production facility."""
    return ProductionFacilityAgent(
        name="test_facility",
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

def test_facility_initialization(facility: ProductionFacilityAgent, test_config: Dict):
    """Test facility initialization."""
    assert facility.name == "test_facility"
    assert facility.config == test_config
    assert facility.simulation_id == "test_sim"
    assert facility.region == Region.NORTH_AMERICA
    assert facility.capacity == test_config['production_facility']['capacity']['North America']
    assert facility.efficiency == test_config['production_facility']['efficiency']
    assert facility.quality_control == test_config['production_facility']['quality_control']
    assert facility.flexibility == test_config['production_facility']['flexibility']

def test_get_capacity(facility: ProductionFacilityAgent):
    """Test capacity calculation."""
    assert facility.get_capacity() == 200  # Initial capacity
    
    # Add an order
    order = Order(
        id="TEST_ORDER_001",
        product_type="Standard",
        quantity=50,
        source_region=Region.NORTH_AMERICA,
        destination_region=Region.EUROPE,
        creation_time=datetime.now(),
        expected_delivery_time=datetime.now() + timedelta(days=5),
        status=OrderStatus.NEW,
        current_location=Region.NORTH_AMERICA
    )
    facility.process_order(order, datetime.now())
    
    assert facility.get_capacity() == 150  # Remaining capacity

def test_process_order_success(facility: ProductionFacilityAgent, test_order: Order):
    """Test successful order processing."""
    current_time = datetime.now()
    result = facility.process_order(test_order, current_time)
    
    assert result is True
    assert test_order.id in facility.current_orders
    assert test_order.status == OrderStatus.PRODUCTION
    assert test_order.current_location == facility.region
    assert facility.current_load == test_order.quantity

def test_process_order_capacity_exceeded(facility: ProductionFacilityAgent):
    """Test order processing when capacity is exceeded."""
    large_order = Order(
        id="TEST_ORDER_002",
        product_type="Standard",
        quantity=300,  # Exceeds capacity
        source_region=Region.NORTH_AMERICA,
        destination_region=Region.EUROPE,
        creation_time=datetime.now(),
        expected_delivery_time=datetime.now() + timedelta(days=5),
        status=OrderStatus.NEW,
        current_location=Region.NORTH_AMERICA
    )
    result = facility.process_order(large_order, datetime.now())
    assert result is False
    assert large_order.id not in facility.current_orders

def test_update_completed_orders(facility: ProductionFacilityAgent, test_order: Order):
    """Test order completion."""
    current_time = datetime.now()
    facility.process_order(test_order, current_time)
    facility.production_times[test_order.id] = 0
    
    completed = facility.update(current_time)
    assert len(completed) == 1
    assert completed[0].id == test_order.id
    assert completed[0].status == OrderStatus.READY_FOR_SHIPPING
    assert test_order.id in facility.completed_orders
    assert test_order.id not in facility.current_orders

def test_quality_and_flexibility_scores(facility: ProductionFacilityAgent):
    """Test quality and flexibility score calculations."""
    assert facility.get_quality_score() == 0.9
    assert facility.get_flexibility_score() == 0.7

def test_production_time_calculation(facility: ProductionFacilityAgent, test_order: Order):
    """Test production time calculation."""
    current_time = datetime.now()
    facility.process_order(test_order, current_time)
    production_time = facility.production_times[test_order.id]
    
    # Base time: 2.0
    # Efficiency factor: 1/0.8 = 1.25
    # Size factor: ceil(10/50) = 1
    # Quality factor: 1 + (0.9 * 0.2) = 1.18
    # Flexibility bonus: 1 - (0.7 * 0.3) = 0.79
    # Expected: ceil(2.0 * 1.25 * 1 * 1.18 * 0.79) = 3
    assert production_time == 3

def test_multiple_orders_processing(facility: ProductionFacilityAgent):
    """Test processing multiple orders."""
    orders = [
        Order(
            id=f"TEST_ORDER_{i}",
            product_type="Standard",
            quantity=10,
            source_region=Region.NORTH_AMERICA,
            destination_region=Region.EUROPE,
            creation_time=datetime.now(),
            expected_delivery_time=datetime.now() + timedelta(days=5),
            status=OrderStatus.NEW,
            current_location=Region.NORTH_AMERICA
        )
        for i in range(3)
    ]
    
    for order in orders:
        facility.process_order(order, datetime.now())
    
    assert len(facility.current_orders) == 3
    assert facility.current_load == 30  # 3 orders of 10 units each 