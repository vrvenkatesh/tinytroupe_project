"""Tests for interactions between different agent types."""

import pytest
from datetime import datetime, timedelta
import uuid

from agents.coo import COOAgent
from agents.regional_manager import RegionalManagerAgent
from agents.supplier import SupplierAgent
from models.enums import Region, OrderStatus, TransportationMode
from models.order import Order
from simulation.config import DEFAULT_CONFIG

@pytest.fixture
def simulation_id():
    """Generate a unique simulation ID."""
    return str(uuid.uuid4())

@pytest.fixture
def agents(simulation_id):
    """Create a set of interacting agents."""
    supplier_na = SupplierAgent(
        name="NA_Supplier",
        region=Region.NORTH_AMERICA,
        supplier_type='tier1',
        config=DEFAULT_CONFIG['supplier'],
        simulation_id=simulation_id
    )
    supplier_eu = SupplierAgent(
        name="EU_Supplier",
        region=Region.EUROPE,
        supplier_type='tier1',
        config=DEFAULT_CONFIG['supplier'],
        simulation_id=simulation_id
    )
    rm_na = RegionalManagerAgent(
        name="NA_Manager",
        region=Region.NORTH_AMERICA,
        config=DEFAULT_CONFIG['regional_manager'],
        simulation_id=simulation_id
    )
    rm_eu = RegionalManagerAgent(
        name="EU_Manager",
        region=Region.EUROPE,
        config=DEFAULT_CONFIG['regional_manager'],
        simulation_id=simulation_id
    )
    coo = COOAgent(
        name="Global_COO",
        config=DEFAULT_CONFIG['coo'],
        simulation_id=simulation_id
    )
    
    return {
        'supplier_na': supplier_na,
        'supplier_eu': supplier_eu,
        'rm_na': rm_na,
        'rm_eu': rm_eu,
        'coo': coo
    }

@pytest.fixture
def world_state():
    """Define the world state for testing."""
    return {
        'current_datetime': datetime.now(),
        'risk_exposure': 0.3,
        'cost_pressure': 0.4,
        'NA_risk': 0.2,
        'EU_risk': 0.3,
        'NA_demand': 0.6,
        'EU_demand': 0.5,
        'NA_efficiency': 0.8,
        'EU_efficiency': 0.7
    }

def test_order_flow_interaction(agents, world_state):
    """Test the order processing flow between agents."""
    # Create a test order
    order = Order(
        id=str(uuid.uuid4()),
        product_type="TestProduct",
        quantity=100,
        source_region=Region.NORTH_AMERICA,
        destination_region=Region.EUROPE,
        creation_time=datetime.now(),
        expected_delivery_time=datetime.now() + timedelta(days=10),
        transportation_mode=TransportationMode.AIR
    )
    
    # Add order to world state
    world_state['active_orders'] = [order]
    
    # Test regional manager's order processing
    rm_actions = agents['rm_na'].manage_region(world_state)
    assert rm_actions['supplier_management']['supplier_engagement'] > 0
    
    # Test supplier's operation
    supplier_performance = agents['supplier_na'].operate(world_state)
    assert supplier_performance['reliability'] > 0
    assert supplier_performance['quality'] > 0

def test_cross_regional_coordination(agents, world_state):
    """Test coordination between regional managers."""
    # Test regional managers' inventory management
    na_actions = agents['rm_na'].manage_region(world_state)
    eu_actions = agents['rm_eu'].manage_region(world_state)
    
    assert na_actions['inventory_management']['safety_stock'] > 0
    assert eu_actions['inventory_management']['safety_stock'] > 0

def test_escalation_flow(agents, world_state):
    """Test the escalation process from supplier to regional manager to COO."""
    # Test supplier's operation under risk
    world_state['NA_risk'] = 0.8  # High risk scenario
    supplier_performance = agents['supplier_na'].operate(world_state)
    assert supplier_performance['reliability'] < 0.7  # Lower reliability due to high risk
    
    # Test regional manager's response
    rm_actions = agents['rm_na'].manage_region(world_state)
    assert rm_actions['supplier_management']['supplier_engagement'] < 0.5

def test_collaborative_decision_making(agents, world_state):
    """Test collaborative decision making between agents."""
    # Test regional managers' transportation management
    na_actions = agents['rm_na'].manage_region(world_state)
    eu_actions = agents['rm_eu'].manage_region(world_state)
    
    assert 'mode_selection' in na_actions['transportation_management']
    assert 'mode_selection' in eu_actions['transportation_management']

def test_error_handling_interaction(agents, world_state):
    """Test how agents handle invalid orders and errors."""
    # Create an invalid order
    invalid_order = Order(
        id=str(uuid.uuid4()),
        product_type="TestProduct",
        quantity=-100,  # Invalid quantity
        source_region=Region.NORTH_AMERICA,
        destination_region=Region.EUROPE,
        creation_time=datetime.now(),
        expected_delivery_time=datetime.now() - timedelta(days=1),  # Past due date
        transportation_mode=TransportationMode.AIR
    )
    
    # Add invalid order to world state
    world_state['active_orders'] = [invalid_order]
    
    # Test regional manager's handling
    rm_actions = agents['rm_na'].manage_region(world_state)
    assert rm_actions['supplier_management']['supplier_engagement'] > 0  # Should still function 