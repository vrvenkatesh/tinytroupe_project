"""Tests for Monte Carlo simulation functions."""

import pytest
import numpy as np

from simulation.monte_carlo import run_monte_carlo_simulation
from simulation.world import create_simulation_world
from simulation.config import DEFAULT_CONFIG

@pytest.fixture
def test_config():
    """Create a smaller configuration for testing."""
    config = DEFAULT_CONFIG.copy()
    config.update({
        'simulation': {
            'monte_carlo_iterations': 5,  # Reduced from 100
            'time_steps': 10,  # Reduced from 365
            'suppliers_per_region': 2,  # Reduced from 3
            'seed': 42,
            'base_demand': 5  # Base demand for order generation
        },
        'supplier': {
            'diversification_enabled': False,
            'reliability': 0.8,
            'quality_score': 0.9,
            'cost_efficiency': 0.7
        },
        'inventory_management': {
            'dynamic_enabled': False,
            'base_stock_level': 100,
            'safety_stock_factor': 1.5
        },
        'logistics': {
            'flexible_routing_enabled': False,
            'reliability': 0.8,
            'cost_efficiency': 0.7,
            'flexibility': 0.6
        },
        'production_facility': {
            'regional_flexibility_enabled': False,
            'efficiency': 0.8,
            'quality_control': 0.9,
            'flexibility': 0.7,
            'base_production_time': 3
        }
    })
    return config

@pytest.fixture
def simulation_world(test_config):
    """Create a simulation world for testing."""
    return create_simulation_world(test_config)

def test_monte_carlo_baseline(test_config, simulation_world):
    """Test Monte Carlo simulation with baseline configuration."""
    metrics = run_monte_carlo_simulation(
        config=test_config,
        world=simulation_world,
        has_supplier_diversification=False,
        has_dynamic_inventory=False,
        has_flexible_transportation=False,
        has_regional_flexibility=False
    )
    
    # Check that all required metrics are present
    required_metrics = [
        'mean_completion_rate',
        'mean_on_time_delivery_rate',
        'mean_average_delay',
        'std_completion_rate',
        'std_on_time_delivery_rate',
        'std_average_delay',
        'min_completion_rate',
        'max_completion_rate',
        'min_on_time_delivery_rate',
        'max_on_time_delivery_rate',
        'min_average_delay',
        'max_average_delay'
    ]
    
    for metric in required_metrics:
        assert metric in metrics
        assert isinstance(metrics[metric], float)
    
    # Check that metrics are within valid ranges
    assert 0 <= metrics['mean_completion_rate'] <= 1
    assert 0 <= metrics['mean_on_time_delivery_rate'] <= 1
    assert metrics['mean_average_delay'] >= 0
    
    # Check that standard deviations are non-negative
    assert metrics['std_completion_rate'] >= 0
    assert metrics['std_on_time_delivery_rate'] >= 0
    assert metrics['std_average_delay'] >= 0
    
    # Check min/max relationships
    assert metrics['min_completion_rate'] <= metrics['mean_completion_rate'] <= metrics['max_completion_rate']
    assert metrics['min_on_time_delivery_rate'] <= metrics['mean_on_time_delivery_rate'] <= metrics['max_on_time_delivery_rate']
    assert metrics['min_average_delay'] <= metrics['mean_average_delay'] <= metrics['max_average_delay']

def test_monte_carlo_all_features(test_config, simulation_world):
    """Test Monte Carlo simulation with all features enabled."""
    # Create a copy of the config to avoid modifying the original
    config = {
        'simulation': test_config.get('simulation', {}).copy(),
        'supplier': test_config.get('supplier', {}).copy(),
        'inventory_management': test_config.get('inventory_management', {}).copy(),
        'logistics': test_config.get('logistics', {}).copy(),
        'production_facility': test_config.get('production_facility', {}).copy()
    }
    
    metrics = run_monte_carlo_simulation(
        config=config,
        world=simulation_world,
        has_supplier_diversification=True,
        has_dynamic_inventory=True,
        has_flexible_transportation=True,
        has_regional_flexibility=True
    )
    
    # Verify that features were properly enabled
    assert config['supplier']['diversification_enabled']
    assert config['inventory_management']['dynamic_enabled']
    assert config['logistics']['flexible_routing_enabled']
    assert config['production_facility']['regional_flexibility_enabled']
    
    # Verify that feature enabling had meaningful impact
    assert config['supplier']['reliability'] > test_config['supplier']['reliability']
    assert config['supplier']['cost_efficiency'] < test_config['supplier']['cost_efficiency']
    assert config['inventory_management']['safety_stock_factor'] > test_config['inventory_management']['safety_stock_factor']
    assert config['logistics']['flexibility'] > test_config['logistics']['flexibility']
    assert config['production_facility']['flexibility'] > test_config['production_facility']['flexibility']
    
    # Basic metric validation
    assert 0 <= metrics['mean_completion_rate'] <= 1
    assert 0 <= metrics['mean_on_time_delivery_rate'] <= 1
    assert metrics['mean_average_delay'] >= 0

def test_monte_carlo_reproducibility(test_config, simulation_world):
    """Test that Monte Carlo simulation is reproducible with same seed."""
    # Create two copies of the config
    config1 = {
        'simulation': test_config.get('simulation', {}).copy(),
        'supplier': test_config.get('supplier', {}).copy(),
        'inventory_management': test_config.get('inventory_management', {}).copy(),
        'logistics': test_config.get('logistics', {}).copy(),
        'production_facility': test_config.get('production_facility', {}).copy()
    }
    config2 = {
        'simulation': test_config.get('simulation', {}).copy(),
        'supplier': test_config.get('supplier', {}).copy(),
        'inventory_management': test_config.get('inventory_management', {}).copy(),
        'logistics': test_config.get('logistics', {}).copy(),
        'production_facility': test_config.get('production_facility', {}).copy()
    }
    
    # Run simulation twice with same seed
    metrics1 = run_monte_carlo_simulation(
        config=config1,
        world=simulation_world
    )
    
    metrics2 = run_monte_carlo_simulation(
        config=config2,
        world=simulation_world
    )
    
    # Results should be identical
    for key in metrics1:
        assert np.isclose(metrics1[key], metrics2[key])
    
    # Change seed and run again
    config3 = config1.copy()
    config3['simulation'] = config1['simulation'].copy()
    config3['simulation']['seed'] = 999
    
    metrics3 = run_monte_carlo_simulation(
        config=config3,
        world=simulation_world
    )
    
    # Results should be different (at least one metric should differ)
    differences = [not np.isclose(metrics1[key], metrics3[key]) for key in metrics1]
    assert any(differences), "Results with different seeds should be different"

def test_monte_carlo_feature_comparison(test_config, simulation_world):
    """Test that different feature combinations produce different results."""
    # Create config copies
    baseline_config = {
        'simulation': test_config.get('simulation', {}).copy(),
        'supplier': test_config.get('supplier', {}).copy(),
        'inventory_management': test_config.get('inventory_management', {}).copy(),
        'logistics': test_config.get('logistics', {}).copy(),
        'production_facility': test_config.get('production_facility', {}).copy()
    }
    
    feature_config = {
        'simulation': test_config.get('simulation', {}).copy(),
        'supplier': test_config.get('supplier', {}).copy(),
        'inventory_management': test_config.get('inventory_management', {}).copy(),
        'logistics': test_config.get('logistics', {}).copy(),
        'production_facility': test_config.get('production_facility', {}).copy()
    }
    
    # Baseline (no features)
    baseline_metrics = run_monte_carlo_simulation(
        config=baseline_config,
        world=simulation_world,
        has_supplier_diversification=False,
        has_dynamic_inventory=False,
        has_flexible_transportation=False,
        has_regional_flexibility=False
    )
    
    # All features
    feature_metrics = run_monte_carlo_simulation(
        config=feature_config,
        world=simulation_world,
        has_supplier_diversification=True,
        has_dynamic_inventory=True,
        has_flexible_transportation=True,
        has_regional_flexibility=True
    )
    
    # Results should show improvement with features enabled
    assert feature_metrics['mean_completion_rate'] > baseline_metrics['mean_completion_rate']
    assert feature_metrics['mean_on_time_delivery_rate'] > baseline_metrics['mean_on_time_delivery_rate']
    assert feature_metrics['mean_average_delay'] < baseline_metrics['mean_average_delay']
    
    # Verify that features were properly enabled in the feature config
    assert feature_config['supplier']['diversification_enabled']
    assert feature_config['inventory_management']['dynamic_enabled']
    assert feature_config['logistics']['flexible_routing_enabled']
    assert feature_config['production_facility']['regional_flexibility_enabled']
    
    # Verify that baseline config remains unchanged
    assert not baseline_config['supplier'].get('diversification_enabled', False)
    assert not baseline_config['inventory_management'].get('dynamic_enabled', False)
    assert not baseline_config['logistics'].get('flexible_routing_enabled', False)
    assert not baseline_config['production_facility'].get('regional_flexibility_enabled', False) 