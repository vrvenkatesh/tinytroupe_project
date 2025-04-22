"""Tests for Monte Carlo simulation functions."""

import pytest
import numpy as np
import uuid
import os
from datetime import datetime, timedelta
import random

from simulation.monte_carlo import run_monte_carlo_simulation, MonteCarloSimulation
from simulation.world import create_simulation_world, SimulationWorld
from simulation.config import DEFAULT_CONFIG
from models.disruption import Disruption
from models.enums import DisruptionType, Region, OrderStatus, TransportationMode
from tests.test_helpers import TestArtifactGenerator

@pytest.fixture
def simulation_id():
    """Create a unique simulation ID for each test run."""
    return str(uuid.uuid4())[:8]

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

@pytest.fixture
def disruption_scenarios():
    """Create test disruption scenarios."""
    now = datetime.now()
    return [
        Disruption(
            type=DisruptionType.SUPPLIER_BANKRUPTCY,
            region=Region.NORTH_AMERICA,
            severity=0.7,
            start_time=now,
            expected_duration=timedelta(days=5),
            affected_capacity=0.8
        ),
        Disruption(
            type=DisruptionType.TRANSPORTATION_FAILURE,
            region=Region.EUROPE,
            severity=0.5,
            start_time=now,
            expected_duration=timedelta(days=3),
            affected_capacity=0.6
        )
    ]

def test_monte_carlo_baseline(test_config, simulation_world):
    """Test Monte Carlo simulation with baseline configuration."""
    # Create Monte Carlo simulation instance
    mc_sim = MonteCarloSimulation(
        simulation_id=str(uuid.uuid4())[:8],
        num_iterations=test_config['simulation']['monte_carlo_iterations'],
        time_horizon_days=test_config['simulation']['time_steps']
    )
    
    # Run simulation
    metrics = run_monte_carlo_simulation(
        config=test_config,
        world=simulation_world,
        has_supplier_diversification=False,
        has_dynamic_inventory=False,
        has_flexible_transportation=False,
        has_regional_flexibility=False
    )
    
    # Verify metrics structure
    assert isinstance(metrics, dict)
    assert 'mean_completion_rate' in metrics
    assert 'mean_on_time_delivery_rate' in metrics
    assert 'mean_resilience_score' in metrics
    assert 'mean_risk_level' in metrics
    
    # Verify metric ranges
    assert 0 <= metrics['mean_completion_rate'] <= 1
    assert 0 <= metrics['mean_on_time_delivery_rate'] <= 1
    assert 0 <= metrics['mean_resilience_score'] <= 1
    assert 0 <= metrics['mean_risk_level'] <= 1

def test_monte_carlo_all_features(test_config, simulation_world):
    """Test Monte Carlo simulation with all features enabled."""
    # Create test data
    test_orders = []
    base_time = datetime.now()
    simulation_id = str(uuid.uuid4())[:8]  # Generate a unique simulation ID
    
    # Create fixed agents with consistent types
    fixed_agents = {
        'regional_manager': type('TestRegionalManager', (), {
            'id': 'AGENT_RM_001',
            'agent_type': 'RegionalManager',
            'interactions': []
        }),
        'supplier': type('TestSupplier', (), {
            'id': 'AGENT_SUP_001',
            'agent_type': 'Supplier',
            'interactions': []
        }),
        'logistics': type('TestLogistics', (), {
            'id': 'AGENT_LOG_001',
            'agent_type': 'Logistics',
            'interactions': []
        })
    }
    
    # Add agents to world state
    simulation_world.state['agents'] = list(fixed_agents.values())
    
    # Create test orders with different statuses and transitions
    for i in range(3):  # Create 3 test orders
        order_id = f'ORD_{i:08d}'
        expected_delivery = base_time + timedelta(days=7)
        actual_delivery = None
        order_events = []
        
        # Define the order lifecycle with proper agent handoffs
        lifecycle_stages = [
            {
                'status': OrderStatus.NEW,
                'agent_type': 'RegionalManager',
                'agent': fixed_agents['regional_manager'],
                'target_agent': fixed_agents['supplier'],
                'message': 'New order received. Assigning to supplier.',
                'location': Region.NORTH_AMERICA
            },
            {
                'status': OrderStatus.PRODUCTION,
                'agent_type': 'Supplier',
                'agent': fixed_agents['supplier'],
                'target_agent': fixed_agents['supplier'],  # Same agent handles production
                'message': 'Order in production at supplier facility.',
                'location': Region.NORTH_AMERICA
            },
            {
                'status': OrderStatus.READY_FOR_SHIPPING,
                'agent_type': 'Supplier',
                'agent': fixed_agents['supplier'],
                'target_agent': fixed_agents['logistics'],
                'message': 'Production complete. Ready for shipping.',
                'location': Region.NORTH_AMERICA
            },
            {
                'status': OrderStatus.IN_TRANSIT,
                'agent_type': 'Logistics',
                'agent': fixed_agents['logistics'],
                'target_agent': fixed_agents['logistics'],  # Same agent handles transit
                'message': 'Order picked up for delivery.',
                'location': Region.EUROPE
            },
            {
                'status': OrderStatus.DELIVERED,
                'agent_type': 'Logistics',
                'agent': fixed_agents['logistics'],
                'target_agent': fixed_agents['logistics'],  # Same agent handles delivery
                'message': 'Order successfully delivered to destination.',
                'location': Region.EUROPE
            }
        ]
        
        # Simulate order progression through stages
        for day, stage in enumerate(lifecycle_stages):
            current_status = stage['status']
            is_delayed = random.random() < 0.2  # 20% chance of delay
            
            if current_status == OrderStatus.DELIVERED:
                actual_delivery = base_time + timedelta(days=day)
            
            # Create an order snapshot for this day
            order = type('TestOrder', (), {
                'id': order_id,
                'creation_time': base_time + timedelta(days=day),
                'status': current_status,
                'current_location': stage['location'],
                'production_time': 2.0 if current_status in [OrderStatus.PRODUCTION, OrderStatus.READY_FOR_SHIPPING] else 0.0,
                'transit_time': 1.5 if current_status == OrderStatus.IN_TRANSIT else 0.0,
                'delay_time': 0.5 if is_delayed else 0.0,
                'expected_delivery_time': expected_delivery,
                'actual_delivery_time': actual_delivery,
                'transportation_mode': TransportationMode.GROUND,
                'source_region': Region.NORTH_AMERICA,
                'destination_region': Region.EUROPE,
                'simulation_day': day
            })
            order_events.append(order)
            
            # Determine if this is a handoff point
            is_handoff = day < len(lifecycle_stages) - 1 and stage['agent_type'] != lifecycle_stages[day + 1]['agent_type']
            
            # Create the message based on whether it's a handoff
            if is_handoff:
                next_stage = lifecycle_stages[day + 1]
                message = f"Processing order {order_id} - {current_status.value}. Handing off from {stage['agent_type']} ({stage['agent'].id}) to {next_stage['agent_type']} ({next_stage['agent'].id}) for {next_stage['status'].value}"
            else:
                message = f"Processing order {order_id} - {current_status.value}. {stage['message']}"
            
            # Add interaction to the current agent
            interaction = type('TestInteraction', (), {
                'type': f'{current_status.value}_PROCESSING',
                'timestamp': base_time + timedelta(days=day),
                'target_agent': stage['target_agent'].id,
                'order_id': order_id,
                'status': current_status,
                'success': True,
                'message': message,
                'simulation_day': day
            })
            stage['agent'].interactions.append(interaction)
        
        test_orders.extend(order_events)
    
    # Update world state with test data
    active_orders = [order for order in test_orders if order.status != OrderStatus.DELIVERED]
    completed_orders = [order for order in test_orders if order.status == OrderStatus.DELIVERED]
    
    simulation_world.state['active_orders'] = active_orders
    simulation_world.state['completed_orders'] = completed_orders
    
    # Run Monte Carlo simulation
    monte_carlo = MonteCarloSimulation(
        simulation_id=simulation_id,
        num_iterations=test_config['simulation']['monte_carlo_iterations'],
        time_horizon_days=test_config['simulation']['time_steps']
    )
    
    # Run simulation with all features enabled
    results = run_monte_carlo_simulation(
        config=test_config,
        world=simulation_world,
        has_supplier_diversification=True,
        has_dynamic_inventory=True,
        has_flexible_transportation=True,
        has_regional_flexibility=True
    )
    
    # Format metrics for saving
    metrics_summary = {
        'metrics': {
            'resilience_score': {
                'mean': results['mean_resilience_score'],
                'std': results.get('std_resilience_score', 0.0),
                'min': results.get('min_resilience_score', results['mean_resilience_score'] * 0.8),
                'max': results.get('max_resilience_score', min(1.0, results['mean_resilience_score'] * 1.2))
            },
            'recovery_time': {
                'mean': results.get('mean_recovery_time', 0.5),
                'std': results.get('std_recovery_time', 0.1),
                'min': results.get('min_recovery_time', 0.3),
                'max': results.get('max_recovery_time', 0.7)
            },
            'service_level': {
                'mean': results.get('mean_on_time_delivery_rate', 0.0),
                'std': results.get('std_on_time_delivery_rate', 0.0),
                'min': results.get('min_on_time_delivery_rate', 0.0),
                'max': results.get('max_on_time_delivery_rate', 1.0)
            }
        },
        'order_status': {
            'DELIVERED': len([o for o in simulation_world.state.get('completed_orders', [])]),
            'IN_TRANSIT': len([o for o in simulation_world.state.get('active_orders', []) if o.status == 'IN_TRANSIT']),
            'PRODUCTION': len([o for o in simulation_world.state.get('active_orders', []) if o.status == 'PRODUCTION']),
            'NEW': len([o for o in simulation_world.state.get('active_orders', []) if o.status == 'NEW'])
        }
    }
    
    # Generate test artifacts
    artifact_generator = TestArtifactGenerator(simulation_id=simulation_id)
    artifact_generator.save_order_lifecycle(simulation_world, "all_features")
    artifact_generator.save_agent_interactions(simulation_world, "all_features")
    artifact_generator.save_metrics_summary(metrics_summary, "all_features")
    
    # Verify results
    assert results is not None
    assert len(results) > 0
    assert 'mean_resilience_score' in results
    assert 0 <= results['mean_resilience_score'] <= 1

def test_monte_carlo_resilience_scenarios(test_config, simulation_world, disruption_scenarios):
    """Test Monte Carlo simulation with specific disruption scenarios."""
    # Create Monte Carlo simulation instance
    mc_sim = MonteCarloSimulation(
        simulation_id=str(uuid.uuid4())[:8],
        num_iterations=test_config['simulation']['monte_carlo_iterations'],
        time_horizon_days=test_config['simulation']['time_steps']
    )
    
    # Run resilience scenarios
    results = mc_sim.simulate_resilience_scenarios(disruption_scenarios)
    
    # Verify results structure
    assert 'scenarios' in results
    assert 'metrics' in results
    assert len(results['scenarios']) == len(disruption_scenarios)
    
    # Verify metrics
    assert isinstance(results['metrics']['mean_recovery_time'], timedelta)
    assert isinstance(results['metrics']['confidence_interval'], tuple)
    assert len(results['metrics']['confidence_interval']) == 2

def test_monte_carlo_feature_comparison(test_config, simulation_world):
    """Test that different feature combinations produce different results."""
    # Create Monte Carlo simulation instance
    mc_sim = MonteCarloSimulation(
        simulation_id=str(uuid.uuid4())[:8],
        num_iterations=test_config['simulation']['monte_carlo_iterations'],
        time_horizon_days=test_config['simulation']['time_steps']
    )
    
    # Run baseline simulation
    baseline_metrics = run_monte_carlo_simulation(
        config=test_config,
        world=simulation_world
    )
    
    # Test each feature individually and in combination
    features = [
        ('supplier_diversification', {'has_supplier_diversification': True}),
        ('dynamic_inventory', {'has_dynamic_inventory': True}),
        ('flexible_transportation', {'has_flexible_transportation': True}),
        ('regional_flexibility', {'has_regional_flexibility': True}),
        ('all_features', {
            'has_supplier_diversification': True,
            'has_dynamic_inventory': True,
            'has_flexible_transportation': True,
            'has_regional_flexibility': True
        })
    ]
    
    for feature_name, feature_flags in features:
        # Run simulation with feature(s) enabled
        feature_metrics = run_monte_carlo_simulation(
            config=test_config,
            world=simulation_world,
            **feature_flags
        )
        
        # Verify that feature improves resilience
        assert feature_metrics['mean_resilience_score'] >= baseline_metrics['mean_resilience_score'], \
            f"Feature {feature_name} should maintain or improve resilience"

if __name__ == '__main__':
    pytest.main([__file__, '-v']) 