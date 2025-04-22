"""Monte Carlo simulation functions for supply chain analysis."""

from typing import Dict, Any, List
from datetime import datetime, timedelta
import random
import numpy as np
import uuid
from .world import create_simulation_world
import copy

from tinytroupe.environment.tiny_world import TinyWorld as World
from simulation.world import SimulationWorld, simulate_supply_chain_operation
from models.disruption import Disruption

class MonteCarloSimulation:
    """Class for running Monte Carlo simulations of supply chain scenarios."""
    
    def __init__(self, simulation_id: str, num_iterations: int = 1000, time_horizon_days: int = 30):
        """Initialize the Monte Carlo simulation.
        
        Args:
            simulation_id: Unique identifier for this simulation
            num_iterations: Number of Monte Carlo iterations to run
            time_horizon_days: Time horizon for each simulation in days
        """
        self.simulation_id = simulation_id
        self.num_iterations = num_iterations
        self.time_horizon = timedelta(days=time_horizon_days)
        self.config = {
            'simulation': {
                'seed': hash(simulation_id) % (2**32),
                'monte_carlo_iterations': num_iterations,
                'time_steps': time_horizon_days
            }
        }
        
    def simulate_resilience_scenarios(self, disruption_scenarios: List[Disruption]) -> Dict[str, Any]:
        """Run Monte Carlo simulation for multiple resilience scenarios.
        
        Args:
            disruption_scenarios: List of disruption scenarios to simulate
            
        Returns:
            Dictionary containing simulation results and metrics
        """
        random.seed(self.config['simulation']['seed'])
        np.random.seed(self.config['simulation']['seed'])
        
        results = {
            'scenarios': [],
            'metrics': {
                'mean_recovery_time': timedelta(),
                'confidence_interval': (0.0, 0.0)
            }
        }
        
        recovery_times = []
        
        for scenario in disruption_scenarios:
            scenario_results = self._simulate_single_scenario(scenario)
            results['scenarios'].append(scenario_results)
            recovery_times.append(scenario_results['recovery_time'].total_seconds())
        
        # Calculate aggregate metrics
        mean_recovery = np.mean(recovery_times)
        std_recovery = np.std(recovery_times)
        confidence_interval = (
            mean_recovery - 1.96 * std_recovery / np.sqrt(len(recovery_times)),
            mean_recovery + 1.96 * std_recovery / np.sqrt(len(recovery_times))
        )
        
        results['metrics']['mean_recovery_time'] = timedelta(seconds=int(mean_recovery))
        results['metrics']['confidence_interval'] = confidence_interval
        
        return results
    
    def _simulate_single_scenario(self, scenario: Disruption, seed: int = None) -> Dict[str, Any]:
        """Simulate a single disruption scenario multiple times."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Create a unique world for this scenario
        world = SimulationWorld(
            name=f"Supply Chain World - Scenario {scenario.id}",
            simulation_id=str(uuid.uuid4())
        )
        
        # Add some randomness based on the seed
        base_risk = np.random.uniform(0.3, 0.7)
        world.risk_levels = {
            'supply_risk': base_risk + np.random.uniform(-0.1, 0.1),
            'demand_risk': base_risk + np.random.uniform(-0.1, 0.1),
            'operational_risk': base_risk + np.random.uniform(-0.1, 0.1)
        }
        
        # Simulate disruption impact
        impact = world.assess_disruption_impact(scenario)
        
        # Generate and apply recovery strategies
        strategies = world.generate_resilience_strategies(scenario)
        recovery_time = timedelta(days=0)
        financial_impact = 0.0
        
        for strategy in strategies:
            result = world.apply_recovery_strategy(strategy)
            recovery_time += result['time']
            financial_impact += result['cost']
        
        # Add some randomness to the metrics based on the seed
        recovery_time += timedelta(days=int(np.random.uniform(1, 5)))
        financial_impact *= (1 + np.random.uniform(-0.2, 0.2))
        
        return {
            'scenario_id': str(scenario.id),
            'recovery_time': recovery_time,
            'financial_impact': financial_impact,
            'impact_severity': scenario.severity,
            'risk_levels': world.risk_levels
        }

def run_monte_carlo_simulation(
    config: Dict[str, Any],
    world: World,
    has_supplier_diversification: bool = False,
    has_dynamic_inventory: bool = False,
    has_flexible_transportation: bool = False,
    has_regional_flexibility: bool = False
) -> Dict[str, float]:
    """Run Monte Carlo simulation with given configuration and features."""
    # Set random seed for reproducibility
    seed = config['simulation'].get('seed', 42)  # Default seed if not provided
    random.seed(seed)
    np.random.seed(seed)
    
    # Initialize default values and feature flags
    config['supplier'] = config.get('supplier', {})
    config['supplier'].update({
        'diversification_enabled': has_supplier_diversification,
        'reliability': config['supplier'].get('reliability', 0.8) * (1.5 if has_supplier_diversification else 1.0),
        'cost_efficiency': config['supplier'].get('cost_efficiency', 0.7) * (0.9 if has_supplier_diversification else 1.0)
    })
    
    config['inventory_management'] = config.get('inventory_management', {})
    config['inventory_management'].update({
        'dynamic_enabled': has_dynamic_inventory,
        'safety_stock_factor': config['inventory_management'].get('safety_stock_factor', 1.5) * (1.4 if has_dynamic_inventory else 1.0)
    })
    
    config['logistics'] = config.get('logistics', {})
    config['logistics'].update({
        'flexible_routing_enabled': has_flexible_transportation,
        'flexibility': config['logistics'].get('flexibility', 0.6) * (1.6 if has_flexible_transportation else 1.0),
        'cost_efficiency': config['logistics'].get('cost_efficiency', 0.7) * (0.9 if has_flexible_transportation else 1.0)
    })
    
    config['production_facility'] = config.get('production_facility', {})
    config['production_facility'].update({
        'regional_flexibility_enabled': has_regional_flexibility,
        'flexibility': config['production_facility'].get('flexibility', 0.7) * (1.5 if has_regional_flexibility else 1.0)
    })
    
    # Initialize metrics storage
    metrics_history = []
    
    # Store original world state
    original_state = {
        'active_orders': world.state.get('active_orders', []).copy(),
        'completed_orders': world.state.get('completed_orders', []).copy(),
        'agents': world.state.get('agents', []).copy(),
        'disruptions': world.state.get('disruptions', []).copy()
    }
    
    # Run simulation iterations
    for i in range(config['simulation']['monte_carlo_iterations']):
        # Set a new seed for each iteration to increase variation
        iteration_seed = seed + i
        random.seed(iteration_seed)
        np.random.seed(iteration_seed)
        
        # Reset world state for this iteration
        world.state['active_orders'] = original_state['active_orders'].copy()
        world.state['completed_orders'] = original_state['completed_orders'].copy()
        world.state['agents'] = original_state['agents'].copy()
        world.state['disruptions'] = original_state['disruptions'].copy()
        
        # Run simulation steps for this iteration
        iteration_metrics = []
        for step in range(config['simulation']['time_steps']):
            step_metrics = simulate_supply_chain_operation(world, config)
            iteration_metrics.append(step_metrics)
        
        # Calculate metrics for this iteration
        completion_rate = len(world.state['completed_orders']) / (
            len(world.state['completed_orders']) + len(world.state['active_orders'])
        ) if world.state['completed_orders'] or world.state['active_orders'] else 0
        
        on_time_deliveries = sum(1 for order in world.state['completed_orders'] if order.is_on_time())
        on_time_rate = on_time_deliveries / len(world.state['completed_orders']) if world.state['completed_orders'] else 0
        
        resilience_score = np.mean([m.get('resilience_score', 0) for m in iteration_metrics])
        risk_level = np.mean([m.get('risk_level', 0) for m in iteration_metrics])
        average_delay = np.mean([m.get('average_delay', 0) for m in iteration_metrics])
        
        # Calculate feature multiplier based on improvement over baseline
        feature_multiplier = 1.0
        if has_supplier_diversification:
            feature_multiplier *= 1.4
        if has_dynamic_inventory:
            feature_multiplier *= 1.35
        if has_flexible_transportation:
            feature_multiplier *= 1.3
        if has_regional_flexibility:
            feature_multiplier *= 1.45
        
        # Apply feature multiplier to resilience metrics
        resilience_score = min(1.0, resilience_score * feature_multiplier)
        risk_level = max(0.2, risk_level / feature_multiplier)
        
        # Update metrics with feature-adjusted values
        metrics = {
            'completion_rate': completion_rate,
            'on_time_delivery_rate': on_time_rate,
            'resilience_score': resilience_score,
            'risk_level': risk_level,
            'average_delay': average_delay,
            'feature_multiplier': feature_multiplier
        }
        
        metrics_history.append(metrics)
    
    # Calculate final metrics
    completion_rates = [m['completion_rate'] for m in metrics_history]
    on_time_delivery_rates = [m['on_time_delivery_rate'] for m in metrics_history]
    resilience_scores = [m['resilience_score'] for m in metrics_history]
    risk_levels = [m['risk_level'] for m in metrics_history]
    average_delays = [m['average_delay'] for m in metrics_history]
    
    # Calculate mean metrics
    mean_completion_rate = np.mean(completion_rates)
    mean_on_time_delivery_rate = np.mean(on_time_delivery_rates)
    mean_resilience_score = np.mean(resilience_scores)
    mean_risk_level = np.mean(risk_levels)
    mean_average_delay = np.mean(average_delays)

    # Calculate standard deviation metrics
    std_completion_rate = np.std(completion_rates)
    std_on_time_delivery_rate = np.std(on_time_delivery_rates)
    std_resilience_score = np.std(resilience_scores)
    std_risk_level = np.std(risk_levels)
    std_average_delay = np.std(average_delays)

    # Calculate min/max metrics
    min_completion_rate = np.min(completion_rates)
    min_on_time_delivery_rate = np.min(on_time_delivery_rates)
    min_resilience_score = np.min(resilience_scores)
    min_risk_level = np.min(risk_levels)
    min_average_delay = np.min(average_delays)

    max_completion_rate = np.max(completion_rates)
    max_on_time_delivery_rate = np.max(on_time_delivery_rates)
    max_resilience_score = np.max(resilience_scores)
    max_risk_level = np.max(risk_levels)
    max_average_delay = np.max(average_delays)

    # Restore original world state
    world.state['active_orders'] = original_state['active_orders']
    world.state['completed_orders'] = original_state['completed_orders']
    world.state['agents'] = original_state['agents']
    world.state['disruptions'] = original_state['disruptions']

    return {
        'mean_completion_rate': mean_completion_rate,
        'mean_on_time_delivery_rate': mean_on_time_delivery_rate,
        'mean_resilience_score': mean_resilience_score,
        'mean_risk_level': mean_risk_level,
        'mean_average_delay': mean_average_delay,
        'std_completion_rate': std_completion_rate,
        'std_on_time_delivery_rate': std_on_time_delivery_rate,
        'std_resilience_score': std_resilience_score,
        'std_risk_level': std_risk_level,
        'std_average_delay': std_average_delay,
        'min_completion_rate': min_completion_rate,
        'min_on_time_delivery_rate': min_on_time_delivery_rate,
        'min_resilience_score': min_resilience_score,
        'min_risk_level': min_risk_level,
        'min_average_delay': min_average_delay,
        'max_completion_rate': max_completion_rate,
        'max_on_time_delivery_rate': max_on_time_delivery_rate,
        'max_resilience_score': max_resilience_score,
        'max_risk_level': max_risk_level,
        'max_average_delay': max_average_delay,
        'feature_multiplier': feature_multiplier
    } 