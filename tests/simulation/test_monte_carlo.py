"""Tests for Monte Carlo simulation functions."""

import pytest
import numpy as np

from simulation.monte_carlo import run_monte_carlo_simulation
from simulation.world import create_simulation_world
from simulation.config import DEFAULT_CONFIG

@pytest.fixture
def simulation_world():
    """Create a simulation world for testing."""
    return create_simulation_world(DEFAULT_CONFIG)

def test_monte_carlo_baseline(simulation_world):
    """Test Monte Carlo simulation with baseline configuration."""
    metrics = run_monte_carlo_simulation(
        config=DEFAULT_CONFIG,
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

def test_monte_carlo_all_features(simulation_world):
    """Test Monte Carlo simulation with all features enabled."""
    metrics = run_monte_carlo_simulation(
        config=DEFAULT_CONFIG,
        world=simulation_world,
        has_supplier_diversification=True,
        has_dynamic_inventory=True,
        has_flexible_transportation=True,
        has_regional_flexibility=True
    )
    
    # Check that features were properly enabled in config
    assert DEFAULT_CONFIG['supplier']['diversification_enabled']
    assert DEFAULT_CONFIG['inventory_management']['dynamic_enabled']
    assert DEFAULT_CONFIG['logistics']['flexible_routing_enabled']
    assert DEFAULT_CONFIG['production_facility']['regional_flexibility_enabled']
    
    # Basic metric validation
    assert 0 <= metrics['mean_completion_rate'] <= 1
    assert 0 <= metrics['mean_on_time_delivery_rate'] <= 1
    assert metrics['mean_average_delay'] >= 0

def test_monte_carlo_reproducibility(simulation_world):
    """Test that Monte Carlo simulation is reproducible with same seed."""
    # Run simulation twice with same seed
    metrics1 = run_monte_carlo_simulation(
        config=DEFAULT_CONFIG,
        world=simulation_world
    )
    
    metrics2 = run_monte_carlo_simulation(
        config=DEFAULT_CONFIG,
        world=simulation_world
    )
    
    # Results should be identical
    for key in metrics1:
        assert np.isclose(metrics1[key], metrics2[key])
    
    # Change seed and run again
    config_different_seed = DEFAULT_CONFIG.copy()
    config_different_seed['simulation']['seed'] = 999
    
    metrics3 = run_monte_carlo_simulation(
        config=config_different_seed,
        world=simulation_world
    )
    
    # Results should be different
    assert not all(np.isclose(metrics1[key], metrics3[key]) for key in metrics1)

def test_monte_carlo_feature_comparison(simulation_world):
    """Test that different feature combinations produce different results."""
    # Baseline
    baseline_metrics = run_monte_carlo_simulation(
        config=DEFAULT_CONFIG,
        world=simulation_world,
        has_supplier_diversification=False,
        has_dynamic_inventory=False,
        has_flexible_transportation=False,
        has_regional_flexibility=False
    )
    
    # All features
    all_features_metrics = run_monte_carlo_simulation(
        config=DEFAULT_CONFIG,
        world=simulation_world,
        has_supplier_diversification=True,
        has_dynamic_inventory=True,
        has_flexible_transportation=True,
        has_regional_flexibility=True
    )
    
    # Results should be different
    assert not np.isclose(
        baseline_metrics['mean_completion_rate'],
        all_features_metrics['mean_completion_rate']
    )
    assert not np.isclose(
        baseline_metrics['mean_on_time_delivery_rate'],
        all_features_metrics['mean_on_time_delivery_rate']
    ) 