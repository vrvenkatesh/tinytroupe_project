"""Tests for supply chain resilience scenarios and strategies."""

import pytest
from datetime import datetime, timedelta
import uuid
import random

from simulation.world import World, create_simulation_world
from simulation.monte_carlo import MonteCarloSimulation
from models.enums import Region, DisruptionType, RiskLevel
from models.disruption import Disruption
from models.resilience_strategy import ResilienceStrategy

@pytest.fixture
def simulation_setup():
    """Create a basic simulation setup for resilience testing."""
    simulation_id = str(uuid.uuid4())
    world = create_simulation_world(config={
        'simulation': {
            'seed': 42,
            'monte_carlo_iterations': 10,
            'suppliers_per_region': 3,
            'time_steps': 30,
            'base_demand': 10
        },
        'coo': {
            'risk_aversion': 0.7,
            'cost_sensitivity': 0.5,
            'strategic_vision': 0.8
        },
        'regional_manager': {
            'local_expertise': 0.8,
            'adaptability': 0.7,
            'communication_skills': 0.6,
            'cost_sensitivity': 0.6
        },
        'supplier': {
            'reliability': 0.8,
            'quality_score': 0.9,
            'cost_efficiency': 0.7,
            'diversification_enabled': False
        },
        'logistics': {
            'reliability': 0.8,
            'cost_efficiency': 0.7,
            'flexibility': 0.6,
            'flexible_routing_enabled': False
        },
        'production_facility': {
            'efficiency': 0.8,
            'quality_control': 0.9,
            'flexibility': 0.7,
            'regional_flexibility_enabled': False,
            'base_production_time': 3
        },
        'inventory_management': {
            'base_stock_level': 100,
            'safety_stock_factor': 1.5,
            'dynamic_enabled': False
        },
        'external_events': {
            'weather': {'frequency': 0.1, 'severity': 0.5},
            'geopolitical': {'frequency': 0.05, 'severity': 0.7},
            'market': {'frequency': 0.2, 'severity': 0.4}
        }
    }, simulation_id=simulation_id)
    
    monte_carlo = MonteCarloSimulation(
        simulation_id=simulation_id,
        num_iterations=10,
        time_horizon_days=30
    )
    return {'world': world, 'monte_carlo': monte_carlo}

@pytest.fixture
def disruption_scenarios():
    """Create various disruption scenarios for testing."""
    current_time = datetime.now()
    return [
        Disruption(
            type=DisruptionType.NATURAL_DISASTER,
            region=Region.NORTH_AMERICA,
            severity=0.8,
            start_time=current_time,
            expected_duration=timedelta(days=5),
            affected_capacity=0.6
        ),
        Disruption(
            type=DisruptionType.SUPPLIER_BANKRUPTCY,
            region=Region.EUROPE,
            severity=0.9,
            start_time=current_time,
            expected_duration=timedelta(days=30),
            affected_capacity=0.8
        ),
        Disruption(
            type=DisruptionType.POLITICAL_UNREST,
            region=Region.EAST_ASIA,
            severity=0.7,
            start_time=current_time,
            expected_duration=timedelta(days=15),
            affected_capacity=0.4
        )
    ]

def test_disruption_impact_assessment(simulation_setup, disruption_scenarios):
    """Test the system's ability to assess disruption impacts."""
    world = simulation_setup['world']
    
    for disruption in disruption_scenarios:
        impact = world.assess_disruption_impact(disruption)
        
        assert 'financial_impact' in impact
        assert 'operational_impact' in impact
        assert 'recovery_time' in impact
        assert isinstance(impact['financial_impact'], float)
        assert 0 <= impact['financial_impact'] <= 1
        assert isinstance(impact['recovery_time'], timedelta)

def test_resilience_strategy_generation(simulation_setup, disruption_scenarios):
    """Test the generation of resilience strategies for different scenarios."""
    world = simulation_setup['world']
    
    for disruption in disruption_scenarios:
        strategies = world.generate_resilience_strategies(disruption)
        
        assert len(strategies) > 0
        for strategy in strategies:
            assert isinstance(strategy, ResilienceStrategy)
            assert strategy.cost >= 0
            assert 0 <= strategy.effectiveness <= 1
            assert strategy.implementation_time > timedelta()
            assert strategy.risk_level in RiskLevel

def test_monte_carlo_resilience_simulation(simulation_setup, disruption_scenarios):
    """Test Monte Carlo simulation for resilience analysis."""
    monte_carlo = simulation_setup['monte_carlo']
    
    results = monte_carlo.simulate_resilience_scenarios(disruption_scenarios)
    
    assert 'scenarios' in results
    assert 'metrics' in results
    assert len(results['scenarios']) == len(disruption_scenarios)
    assert 'mean_recovery_time' in results['metrics']
    assert 'confidence_interval' in results['metrics']

def test_adaptive_response_mechanism(simulation_setup):
    """Test the system's adaptive response to changing conditions."""
    world = simulation_setup['world']
    
    # Initial state
    initial_state = world.get_current_state()
    initial_resilience = initial_state['metrics']['resilience_score']
    initial_risks = {k: v for k, v in initial_state['risk_levels'].items()}
    
    # Simulate changing conditions
    for _ in range(5):
        # Update with random risks
        new_risks = {
            'supply_risk': random.uniform(0.2, 0.8),
            'demand_risk': random.uniform(0.2, 0.8),
            'operational_risk': random.uniform(0.2, 0.8)
        }
        world.update_risk_levels(new_risks)
        
        # Generate and apply response
        response = world.generate_adaptive_response()
        assert 'actions' in response
        assert 'expected_impact' in response
        
        # Store state before applying response
        pre_response_state = world.get_current_state()
        world.apply_adaptive_response(response)
        post_response_state = world.get_current_state()
        
        # Verify that the response had an impact
        assert post_response_state != pre_response_state
        assert len(post_response_state['metrics']['risk_exposure_trend']) > len(pre_response_state['metrics']['risk_exposure_trend'])
        
        # Verify risk levels changed
        for risk_type in post_response_state['risk_levels']:
            assert post_response_state['risk_levels'][risk_type] <= pre_response_state['risk_levels'][risk_type]
    
    # Final state
    final_state = world.get_current_state()
    
    # Verify overall changes
    assert final_state != initial_state
    assert len(final_state['metrics']['risk_exposure_trend']) > len(initial_state['metrics']['risk_exposure_trend'])
    assert final_state['metrics']['resilience_score'] > initial_resilience
    
    # Verify risk levels improved
    final_avg_risk = sum(final_state['risk_levels'].values()) / len(final_state['risk_levels'])
    initial_avg_risk = sum(initial_risks.values()) / len(initial_risks)
    assert final_avg_risk < initial_avg_risk

def test_recovery_strategy_effectiveness(simulation_setup, disruption_scenarios):
    """Test the effectiveness of recovery strategies."""
    world = simulation_setup['world']
    disruption = disruption_scenarios[0]
    
    # Generate and apply recovery strategy
    strategy = world.generate_recovery_strategy(disruption)
    initial_impact = world.assess_disruption_impact(disruption)
    
    world.apply_recovery_strategy(strategy)
    post_recovery_impact = world.assess_disruption_impact(disruption)
    
    assert post_recovery_impact['financial_impact'] < initial_impact['financial_impact']
    assert post_recovery_impact['recovery_time'] < initial_impact['recovery_time']

def test_multi_region_resilience(simulation_setup):
    """Test resilience mechanisms across multiple regions."""
    world = simulation_setup['world']
    
    # Simulate multi-region scenario
    scenario = {
        Region.NORTH_AMERICA: {'risk': 0.7, 'capacity': 0.6},
        Region.EUROPE: {'risk': 0.4, 'capacity': 0.8},
        Region.EAST_ASIA: {'risk': 0.5, 'capacity': 0.7}
    }
    
    response = world.optimize_multi_region_resilience(scenario)
    
    assert 'regional_strategies' in response
    assert len(response['regional_strategies']) == len(scenario)
    assert 'global_impact' in response
    
    for region in scenario:
        assert region in response['regional_strategies']
        region_strategy = response['regional_strategies'][region]
        assert 'capacity_adjustment' in region_strategy
        assert 'risk_mitigation' in region_strategy

def test_resilience_metrics_tracking(simulation_setup):
    """Test the tracking of resilience metrics over time."""
    world = simulation_setup['world']
    
    # Initialize tracking
    tracking_start = datetime.now()
    tracking_period = timedelta(days=10)
    
    metrics = world.track_resilience_metrics(
        start_time=tracking_start,
        duration=tracking_period
    )
    
    assert 'time_series' in metrics
    assert len(metrics['time_series']) > 0
    assert 'average_recovery_time' in metrics
    assert 'risk_exposure_trend' in metrics
    assert 'resilience_score' in metrics
    
    # Verify metric values
    assert 0 <= metrics['resilience_score'] <= 1
    assert isinstance(metrics['average_recovery_time'], timedelta)
    assert all(0 <= score <= 1 for score in metrics['risk_exposure_trend']) 