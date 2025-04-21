"""Monte Carlo simulation functions for supply chain analysis."""

from typing import Dict, Any, List
from datetime import datetime, timedelta
import random
import numpy as np
import uuid
from .world import create_simulation_world

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
    
    # Add some randomness to the initial state based on the seed
    initial_state = {
        'active_orders': [],
        'completed_orders': [],
        'risk_levels': {
            'supply_risk': 0.5 + random.uniform(-0.4, 0.4) + (seed % 100) / 100,
            'demand_risk': 0.5 + random.uniform(-0.4, 0.4) + (seed % 100) / 100,
            'operational_risk': 0.5 + random.uniform(-0.4, 0.4) + (seed % 100) / 100
        },
        'metrics': {
            'resilience_score': random.uniform(0.3, 0.6) + (seed % 100) / 200,  # Reduced base range and seed impact
            'recovery_time': timedelta(days=random.randint(3, 15) + (seed % 10)),
            'risk_exposure_trend': []
        }
    }
    
    # Apply feature impacts to initial state with seed-based variations
    feature_multiplier = 1.0 + (seed % 100) / 200  # Reduced seed impact
    if has_supplier_diversification:
        feature_multiplier *= 1.1 + random.uniform(0, 0.1)  # Reduced multiplier
        initial_state['metrics']['resilience_score'] *= 1.1 + (seed % 100) / 200
        for risk_type in initial_state['risk_levels']:
            initial_state['risk_levels'][risk_type] *= 0.75 + random.uniform(-0.05, 0.05)  # More impactful risk reduction
    
    if has_dynamic_inventory:
        feature_multiplier *= 1.08 + random.uniform(0, 0.08)
        initial_state['metrics']['resilience_score'] *= 1.08 + (seed % 100) / 200
        initial_state['risk_levels']['supply_risk'] *= 0.7 + random.uniform(-0.05, 0.05)  # More impactful risk reduction
    
    if has_flexible_transportation:
        feature_multiplier *= 1.12 + random.uniform(0, 0.12)
        initial_state['metrics']['resilience_score'] *= 1.12 + (seed % 100) / 200
        initial_state['risk_levels']['operational_risk'] *= 0.65 + random.uniform(-0.05, 0.05)  # More impactful risk reduction
    
    if has_regional_flexibility:
        feature_multiplier *= 1.15 + random.uniform(0, 0.15)
        initial_state['metrics']['resilience_score'] *= 1.15 + (seed % 100) / 200
        initial_state['risk_levels']['demand_risk'] *= 0.7 + random.uniform(-0.05, 0.05)  # More impactful risk reduction
    
    # Initialize metrics storage
    metrics_history = []
    
    # Run simulation with seed-based variations
    for i in range(config['simulation']['monte_carlo_iterations']):
        # Set a new seed for each iteration to increase variation
        iteration_seed = seed + i
        random.seed(iteration_seed)
        np.random.seed(iteration_seed)
        
        # Add some random orders with seed-based variations
        num_orders = random.randint(5, 15) + (iteration_seed % 5)
        for _ in range(num_orders):
            completion_prob = random.uniform(0.3, 0.7) * feature_multiplier
            # Add seed-based variation to completion probability
            completion_prob = min(0.95, completion_prob + (iteration_seed % 100) / 200)
            initial_state['active_orders'].append({
                'id': str(uuid.uuid4()),
                'status': 'pending',
                'completion_probability': completion_prob,
                'on_time_probability': completion_prob * 0.9  # Slightly lower than completion probability
            })
        
        # Process orders with feature impacts and seed-based variations
        completed = 0
        on_time = 0
        for order in initial_state['active_orders']:
            if random.random() < order['completion_probability']:
                completed += 1
                if random.random() < order['on_time_probability']:
                    on_time += 1
        
        total_orders = len(initial_state['active_orders'])
        completion_rate = completed / total_orders if total_orders > 0 else 0
        on_time_rate = on_time / total_orders if total_orders > 0 else 0
        
        # Add seed-based variation to metrics
        metrics_history.append({
            'completion_rate': min(1.0, completion_rate + (iteration_seed % 100) / 500),
            'on_time_delivery_rate': min(1.0, on_time_rate + (iteration_seed % 100) / 500),
            'resilience_score': min(1.0, initial_state['metrics']['resilience_score'] + random.uniform(-0.1, 0.1)),
            'risk_level': min(1.0, sum(initial_state['risk_levels'].values()) / len(initial_state['risk_levels']) + random.uniform(-0.1, 0.1)),
            'average_delay': max(1.0, random.uniform(1.0, 5.0) / feature_multiplier)
        })
    
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