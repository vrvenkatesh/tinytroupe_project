"""Tests for the Supplier Agent."""

import pytest
from datetime import datetime
from typing import Dict

from agents.supplier import SupplierAgent
from models.enums import Region

@pytest.fixture
def test_config() -> Dict:
    """Create a test configuration."""
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
        }
    }

@pytest.fixture
def supplier(test_config: Dict) -> SupplierAgent:
    """Create a test supplier."""
    return SupplierAgent(
        name="test_supplier",
        config=test_config,
        simulation_id="test_sim",
        region=Region.NORTH_AMERICA,
        supplier_type="tier1"
    )

@pytest.fixture
def world_state() -> Dict:
    """Create a test world state."""
    return {
        'North America_risk': 0.3,
        'North America_quality': 0.8,
        'North America_cost': 0.4,
        'North America_capacity': 0.7,
        'current_datetime': datetime.now()
    }

def test_supplier_initialization(supplier: SupplierAgent, test_config: Dict):
    """Test supplier initialization."""
    assert supplier.name == "test_supplier"
    assert supplier.config == test_config
    assert supplier.simulation_id == "test_sim"
    assert supplier.region == Region.NORTH_AMERICA
    assert supplier.supplier_type == "tier1"

def test_reliability_calculation(supplier: SupplierAgent, world_state: Dict):
    """Test reliability calculation."""
    reliability = supplier._calculate_reliability(world_state)
    
    # Base reliability: 0.8
    # Regional risk: 0.3
    # Formula: base_reliability * (1.0 - regional_risk * 0.5)
    # Expected: 0.8 * (1.0 - 0.3 * 0.5) = 0.8 * 0.85 = 0.68
    assert abs(reliability - 0.68) < 0.01

def test_quality_calculation(supplier: SupplierAgent, world_state: Dict):
    """Test quality calculation."""
    quality = supplier._calculate_quality(world_state)
    
    # Base quality: 0.9
    # Regional quality: 0.8
    # Formula: base_quality * (0.5 + regional_quality * 0.5)
    # Expected: 0.9 * (0.5 + 0.8 * 0.5) = 0.9 * 0.9 = 0.81
    assert abs(quality - 0.81) < 0.01

def test_cost_calculation(supplier: SupplierAgent, world_state: Dict):
    """Test cost calculation."""
    cost = supplier._calculate_cost(world_state)
    
    # Base cost efficiency: 0.7
    # Regional cost: 0.4
    # Formula: base_cost * (1.0 - regional_cost * 0.3)
    # Expected: 0.7 * (1.0 - 0.4 * 0.3) = 0.7 * 0.88 = 0.616
    assert abs(cost - 0.616) < 0.01

def test_capacity_calculation(supplier: SupplierAgent, world_state: Dict):
    """Test capacity calculation."""
    capacity = supplier._calculate_capacity(world_state)
    
    # Base capacity: 200 (North America)
    # Regional capacity: 0.7
    # Formula: (base_capacity / 300) * (0.5 + regional_capacity * 0.5)
    # Expected: (200/300) * (0.5 + 0.7 * 0.5) = 0.667 * 0.85 = 0.567
    assert abs(capacity - 0.567) < 0.01

def test_operate_performance(supplier: SupplierAgent, world_state: Dict):
    """Test overall supplier operation."""
    performance = supplier.operate(world_state)
    
    assert isinstance(performance, dict)
    assert all(key in performance for key in ['reliability', 'quality', 'cost', 'capacity'])
    assert 0 <= performance['reliability'] <= 1
    assert 0 <= performance['quality'] <= 1
    assert 0 <= performance['cost'] <= 1
    assert 0 <= performance['capacity'] <= 1

def test_missing_regional_data(supplier: SupplierAgent):
    """Test handling of missing regional data."""
    empty_world_state = {'current_datetime': datetime.now()}
    performance = supplier.operate(empty_world_state)
    
    # Should use default values when regional data is missing
    assert isinstance(performance, dict)
    assert all(key in performance for key in ['reliability', 'quality', 'cost', 'capacity'])
    assert performance['reliability'] == 0.8 * (1.0 - 0.5 * 0.5)  # Using default risk of 0.5
    assert performance['quality'] == 0.9 * (0.5 + 0.5 * 0.5)  # Using default quality of 0.5

def test_different_regions(test_config: Dict):
    """Test supplier behavior in different regions."""
    regions = [Region.NORTH_AMERICA, Region.EUROPE, Region.EAST_ASIA]
    world_state = {
        'current_datetime': datetime.now(),
        'North America_risk': 0.3,
        'Europe_risk': 0.4,
        'East Asia_risk': 0.5
    }
    
    suppliers = [
        SupplierAgent(
            name=f"test_supplier_{region.value}",
            config=test_config,
            simulation_id="test_sim",
            region=region,
            supplier_type="tier1"
        )
        for region in regions
    ]
    
    performances = [supplier.operate(world_state) for supplier in suppliers]
    
    # Verify each supplier has unique performance based on region
    reliability_values = [p['reliability'] for p in performances]
    assert len(set(reliability_values)) == len(regions)  # Each should be unique

def test_supplier_types(test_config: Dict):
    """Test different supplier types."""
    supplier_types = ["tier1", "raw_material", "contract_manufacturer"]
    suppliers = [
        SupplierAgent(
            name=f"test_supplier_{type_}",
            config=test_config,
            simulation_id="test_sim",
            region=Region.NORTH_AMERICA,
            supplier_type=type_
        )
        for type_ in supplier_types
    ]
    
    for supplier in suppliers:
        assert supplier.supplier_type in supplier_types
        performance = supplier.operate({'current_datetime': datetime.now()})
        assert isinstance(performance, dict) 