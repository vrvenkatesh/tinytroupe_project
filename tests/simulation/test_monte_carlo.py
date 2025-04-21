"""Tests for Monte Carlo simulation functions."""

import pytest
import numpy as np
import uuid
import os
from datetime import datetime, timedelta

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
    # Create Monte Carlo simulation instance
    mc_sim = MonteCarloSimulation(
        simulation_id=str(uuid.uuid4())[:8],
        num_iterations=test_config['simulation']['monte_carlo_iterations'],
        time_horizon_days=test_config['simulation']['time_steps']
    )
    
    # Run simulation with all features enabled
    metrics = run_monte_carlo_simulation(
        config=test_config,
        world=simulation_world,
        has_supplier_diversification=True,
        has_dynamic_inventory=True,
        has_flexible_transportation=True,
        has_regional_flexibility=True
    )
    
    # Create test orders that will progress through the lifecycle
    base_time = datetime.now()
    test_orders = []
    num_orders = 5  # Create 5 orders that will progress through different stages
    
    # Define the lifecycle stages and their timing
    lifecycle_stages = [
        (OrderStatus.NEW, 0),  # Day 0: Order is created
        (OrderStatus.PRODUCTION, 1),  # Day 1: Order enters production
        (OrderStatus.READY_FOR_SHIPPING, 3),  # Day 3: Production complete
        (OrderStatus.IN_TRANSIT, 4),  # Day 4: Order starts shipping
        (OrderStatus.DELIVERED, 7)  # Day 7: Order is delivered
    ]
    
    # Create orders at different stages of their lifecycle
    for i in range(num_orders):
        order_events = []
        order_id = f'ORD_{i:08d}'
        
        # Calculate the current stage based on simulation day
        current_day = i % len(lifecycle_stages)  # Spread orders across different stages
        
        for day in range(8):  # Track 8 days of history
            # Find the appropriate status for this day
            current_status = OrderStatus.NEW  # Default status
            for stage, stage_day in lifecycle_stages:
                if day >= stage_day:
                    current_status = stage
            
            # Calculate delivery timing flags
            expected_delivery = base_time + timedelta(days=6)
            current_time = base_time + timedelta(days=day)
            actual_delivery = base_time + timedelta(days=7) if current_status == OrderStatus.DELIVERED else None
            
            is_delayed = False
            is_on_time = True
            
            # Only check delay status if we have an actual delivery time or if we've exceeded expected delivery
            if actual_delivery:
                is_delayed = actual_delivery > expected_delivery
                is_on_time = not is_delayed
            elif current_time > expected_delivery:
                is_delayed = True
                is_on_time = False
            
            # Create an order snapshot for this day
            order = type('TestOrder', (), {
                'id': order_id,
                'creation_time': base_time + timedelta(days=day),
                'status': current_status,
                'current_location': Region.NORTH_AMERICA if current_status in [OrderStatus.NEW, OrderStatus.PRODUCTION, OrderStatus.READY_FOR_SHIPPING] else Region.EUROPE,
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
            
            # Create corresponding agent interactions with specific agent types
            agent_types = ['Supplier', 'Manufacturer', 'Logistics']
            agent_type = agent_types[i % len(agent_types)]
            
            agent = type(f'Test{agent_type}', (), {
                'id': f'AGENT_{i % 3:03d}',  # 3 different agents handling orders
                'agent_type': agent_type,  # Add explicit agent_type attribute
                'interactions': [
                    type('TestInteraction', (), {
                        'type': f'{current_status.value}_PROCESSING',
                        'timestamp': base_time + timedelta(days=day),
                        'target_agent': f'AGENT_{(i + 1) % 3:03d}',  # Interact with next agent
                        'order_id': order_id,
                        'status': current_status,
                        'success': True,
                        'message': f'Processing order {order_id} - {current_status.value}',
                        'simulation_day': day
                    })
                ]
            })
            
            if 'agents' not in simulation_world.state:
                simulation_world.state['agents'] = []
            simulation_world.state['agents'].append(agent)
        
        test_orders.extend(order_events)
    
    # Update world state with test data
    active_orders = [order for order in test_orders if order.status != OrderStatus.DELIVERED]
    completed_orders = [order for order in test_orders if order.status == OrderStatus.DELIVERED]
    
    simulation_world.state['active_orders'] = active_orders
    simulation_world.state['completed_orders'] = completed_orders
    
    # Format metrics for artifact generation
    metrics_formatted = {
        'metrics': {
            'resilience_score': {
                'mean': metrics['mean_resilience_score'],
                'std': metrics['std_resilience_score'],
                'min': metrics.get('min_resilience_score', 0),
                'max': metrics.get('max_resilience_score', 1)
            },
            'service_level': {
                'mean': metrics['mean_completion_rate'],
                'std': metrics['std_completion_rate'],
                'min': metrics.get('min_completion_rate', 0),
                'max': metrics.get('max_completion_rate', 1)
            },
            'recovery_time': {
                'mean': metrics['mean_average_delay'],
                'std': metrics['std_average_delay'],
                'min': metrics.get('min_average_delay', 0),
                'max': metrics.get('max_average_delay', 10)
            },
            'risk_exposure': {
                'mean': metrics['mean_risk_level'],
                'std': metrics['std_risk_level'],
                'min': metrics.get('min_risk_level', 0),
                'max': metrics.get('max_risk_level', 1)
            }
        },
        'order_status': {
            'new': len([o for o in active_orders if o.status == OrderStatus.NEW]),
            'in_production': len([o for o in active_orders if o.status == OrderStatus.PRODUCTION]),
            'ready_for_shipping': len([o for o in active_orders if o.status == OrderStatus.READY_FOR_SHIPPING]),
            'in_transit': len([o for o in active_orders if o.status == OrderStatus.IN_TRANSIT]),
            'delivered': len(completed_orders),
            'delayed': sum(1 for o in test_orders if o.delay_time > 0)
        }
    }
    
    # Generate artifacts only for this test
    artifact_generator = TestArtifactGenerator(simulation_id=str(uuid.uuid4())[:8])
    artifact_generator.generate_artifacts(
        world=simulation_world,
        metrics=metrics_formatted,
        scenario_name="all_features"
    )
    
    # Verify that features improve resilience
    assert metrics['mean_resilience_score'] > 0.5  # Should be better than average
    assert metrics['mean_risk_level'] < 0.5  # Risk should be reduced

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