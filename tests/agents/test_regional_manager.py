"""Tests for the Regional Manager agent."""

import pytest
from datetime import datetime, timedelta
import uuid

from agents.regional_manager import RegionalManagerAgent
from models.enums import Region, OrderStatus
from models.order import Order
from simulation.config import DEFAULT_CONFIG

@pytest.fixture
def regional_manager():
    """Create a Regional Manager agent for testing."""
    return RegionalManagerAgent(
        name="TestManager_NA",
        config=DEFAULT_CONFIG['regional_manager'],
        simulation_id=str(uuid.uuid4()),
        region=Region.NORTH_AMERICA
    )

@pytest.fixture
def sample_world_state():
    """Create a sample world state for testing."""
    current_time = datetime.now()
    
    # Create orders
    orders = [
        Order(
            id=f"test_order_{i}",
            product_type="Standard",
            quantity=10,
            source_region=Region.NORTH_AMERICA,
            destination_region=Region.EUROPE,
            creation_time=current_time,
            expected_delivery_time=current_time + timedelta(days=5)
        )
        for i in range(5)
    ]
    
    # Create mock production facilities
    class MockFacility:
        def __init__(self, name, region):
            self.name = name
            self.region = region
            self.current_orders = []
            self.capacity = 100
    
    facilities = [
        MockFacility(f"Facility_NA_{i}", Region.NORTH_AMERICA)
        for i in range(2)
    ]
    
    return {
        'active_orders': orders,
        'production_facilities': facilities,
        'current_datetime': current_time,
        'NORTH_AMERICA_risk': 0.3,
        'NORTH_AMERICA_cost': 0.4,
        'NORTH_AMERICA_demand': 0.6,
        'NORTH_AMERICA_supply_risk': 0.2,
        'NORTH_AMERICA_infrastructure': 0.8,
        'NORTH_AMERICA_congestion': 0.3,
        'NORTH_AMERICA_efficiency': 0.7,
        'NORTH_AMERICA_flexibility': 0.6
    }

def test_regional_manager_initialization(regional_manager):
    """Test Regional Manager initialization."""
    assert regional_manager.name == "TestManager_NA"
    assert regional_manager.config == DEFAULT_CONFIG['regional_manager']
    assert isinstance(regional_manager.simulation_id, str)
    assert regional_manager.region == Region.NORTH_AMERICA

def test_manage_suppliers(regional_manager, sample_world_state):
    """Test supplier management functionality."""
    result = regional_manager._manage_suppliers(sample_world_state)
    
    assert 'supplier_engagement' in result
    assert 'cost_negotiation' in result
    assert 0 <= result['supplier_engagement'] <= 1
    assert 0 <= result['cost_negotiation'] <= 1

def test_manage_inventory(regional_manager, sample_world_state):
    """Test inventory management functionality."""
    result = regional_manager._manage_inventory(sample_world_state)
    
    assert 'safety_stock' in result
    assert 'dynamic_adjustment' in result
    assert result['safety_stock'] > 0
    assert isinstance(result['dynamic_adjustment'], bool)

def test_manage_transportation(regional_manager, sample_world_state):
    """Test transportation management functionality."""
    result = regional_manager._manage_transportation(sample_world_state)
    
    assert 'mode_selection' in result
    assert 'route_optimization' in result
    assert sum(result['mode_selection'].values()) == pytest.approx(1.0, rel=1e-9)
    assert 0 <= result['route_optimization'] <= 1

def test_manage_production(regional_manager, sample_world_state):
    """Test production management functionality."""
    result = regional_manager._manage_production(sample_world_state)
    
    assert 'efficiency_improvement' in result
    assert 'flexibility_level' in result
    assert 0 <= result['efficiency_improvement'] <= 1
    assert 0 <= result['flexibility_level'] <= 1

def test_custom_act_no_world_state(regional_manager):
    """Test custom_act behavior with no world state."""
    message = regional_manager.custom_act()
    assert "No world state provided" in message

def test_custom_act_with_orders(regional_manager, sample_world_state):
    """Test custom_act behavior with orders."""
    messages = regional_manager.custom_act(sample_world_state, return_actions=True)
    
    assert isinstance(messages, list)
    assert len(messages) > 0
    assert any("Assigned order" in msg for msg in messages)
    
    # Check that orders were processed
    for order in sample_world_state['active_orders']:
        if order.source_region == regional_manager.region:
            assert order.status in [OrderStatus.PRODUCTION, OrderStatus.NEW]
            if order.status == OrderStatus.PRODUCTION:
                assert order.production_facility is not None

def test_custom_act_facility_capacity(regional_manager, sample_world_state):
    """Test handling of facility capacity constraints."""
    # Set facility capacity to 0
    for facility in sample_world_state['production_facilities']:
        facility.capacity = 0
    
    messages = regional_manager.custom_act(sample_world_state, return_actions=True)
    assert any("No production facilities available" in msg for msg in messages)

def test_custom_act_error_handling(regional_manager, sample_world_state):
    """Test error handling in custom_act."""
    # Test with invalid world state
    message = regional_manager.custom_act(None)
    assert "No world state provided" in message
    
    # Test with missing required fields
    invalid_state = {
        'active_orders': [],
        'current_datetime': datetime.now()  # Add required field
    }
    message = regional_manager.custom_act(invalid_state)
    assert "Error in Regional Manager" in message 