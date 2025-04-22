"""Tests for interactions between different agent types."""

import pytest
from datetime import datetime, timedelta
import uuid

from agents.coo import COOAgent
from agents.regional_manager import RegionalManagerAgent
from agents.supplier import SupplierAgent
from models.enums import Region, OrderStatus, TransportationMode
from models.order import Order

@pytest.fixture
def simulation_id():
    """Generate a unique simulation ID."""
    return str(uuid.uuid4())

@pytest.fixture
def test_config():
    """Create test configuration."""
    return {
        'supplier': {
            'reliability': 0.8,
            'quality': 0.9,
            'cost_efficiency': 0.7,
            'capacity': {
                'North America': 200,
                'Europe': 150,
                'East Asia': 300,
                'Southeast Asia': 250,
                'South Asia': 180
            }
        },
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
        'coo': {
            'risk_tolerance': 0.7,
            'cost_sensitivity': 0.8,
            'decision_threshold': 0.6
        },
        'local_expertise': 0.8,
        'cost_sensitivity': 0.7,
        'adaptability': 0.9,
        'dynamic_enabled': True
    }

@pytest.fixture
def agents(simulation_id, test_config):
    """Create a set of interacting agents."""
    supplier_na = SupplierAgent(
        name="NA_Supplier",
        region=Region.NORTH_AMERICA,
        supplier_type='tier1',
        config=test_config,
        simulation_id=simulation_id
    )
    supplier_eu = SupplierAgent(
        name="EU_Supplier",
        region=Region.EUROPE,
        supplier_type='tier1',
        config=test_config,
        simulation_id=simulation_id
    )
    supplier_raw = SupplierAgent(
        name="Raw_Material_Supplier",
        region=Region.NORTH_AMERICA,
        supplier_type='raw_material',
        config=test_config,
        simulation_id=simulation_id
    )
    rm_na = RegionalManagerAgent(
        name="NA_Manager",
        region=Region.NORTH_AMERICA,
        config=test_config,
        simulation_id=simulation_id
    )
    rm_eu = RegionalManagerAgent(
        name="EU_Manager",
        region=Region.EUROPE,
        config=test_config,
        simulation_id=simulation_id
    )
    coo = COOAgent(
        name="Global_COO",
        config=test_config,
        simulation_id=simulation_id
    )
    
    return {
        'supplier_na': supplier_na,
        'supplier_eu': supplier_eu,
        'supplier_raw': supplier_raw,
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
        'EU_efficiency': 0.7,
        'North America_quality': 0.8,
        'Europe_quality': 0.7,
        'North America_capacity': 0.7,
        'Europe_capacity': 0.6,
        'North America_cost': 0.4,
        'Europe_cost': 0.5
    }

@pytest.fixture
def supplier_config():
    return {
        "name": "Test Supplier",
        "cost_efficiency": 0.7,
        "base_quality": {
            "Tier1": 0.85,  # Higher quality for Tier 1
            "Raw_Material": 0.75  # Lower quality for raw materials
        },
        "base_reliability": {
            "North_America": 0.9,  # Higher reliability in NA
            "Europe": 0.8  # Lower reliability in EU
        },
        "engagement_factors": {
            "North_America": 0.8,  # Different engagement factors
            "Europe": 0.6
        }
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
        status=OrderStatus.NEW,
        current_location=Region.NORTH_AMERICA
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
        status=OrderStatus.NEW,
        current_location=Region.NORTH_AMERICA
    )
    
    # Add invalid order to world state
    world_state['active_orders'] = [invalid_order]
    
    # Test regional manager's handling
    rm_actions = agents['rm_na'].manage_region(world_state)
    assert rm_actions['supplier_management']['supplier_engagement'] > 0  # Should still function 

def test_supplier_type_interaction(world_state, supplier_config):
    # Create suppliers with different types
    tier1_supplier = SupplierAgent(
        name="NA_Supplier",
        config=supplier_config,
        simulation_id="test_sim",
        supplier_type="Tier1",
        region="North_America"
    )
    
    raw_supplier = SupplierAgent(
        name="Raw_Material_Supplier",
        config=supplier_config,
        simulation_id="test_sim",
        supplier_type="Raw_Material",
        region="North_America"
    )
    
    # Let both suppliers operate
    tier1_supplier.act(world_state)
    raw_supplier.act(world_state)
    
    # Get performance metrics
    tier1_performance = tier1_supplier.get_performance_metrics()
    raw_performance = raw_supplier.get_performance_metrics()
    
    # Verify different quality levels based on supplier type
    assert abs(tier1_performance['quality'] - raw_performance['quality']) > 0.05
    assert tier1_performance['quality'] > raw_performance['quality']

def test_regional_supplier_coordination(world_state, supplier_config):
    # Create suppliers in different regions
    na_supplier = SupplierAgent(
        name="NA_Supplier",
        config=supplier_config,
        simulation_id="test_sim",
        supplier_type="Tier1",
        region="North_America"
    )
    
    eu_supplier = SupplierAgent(
        name="EU_Supplier",
        config=supplier_config,
        simulation_id="test_sim",
        supplier_type="Tier1",
        region="Europe"
    )
    
    # Let both suppliers operate
    na_supplier.act(world_state)
    eu_supplier.act(world_state)
    
    # Calculate engagement levels
    na_engagement = na_supplier.calculate_engagement_level()
    eu_engagement = eu_supplier.calculate_engagement_level()
    
    # Verify different engagement levels based on region
    assert abs(na_engagement - eu_engagement) > 0.05
    assert na_engagement > eu_engagement  # NA should have higher engagement 