"""Monte Carlo simulation functions for supply chain analysis."""

from typing import Dict, Any
import random
import numpy as np

from tinytroupe.environment.tiny_world import TinyWorld as World
from simulation.world import simulate_supply_chain_operation

def run_monte_carlo_simulation(
    config: Dict[str, Any],
    world: World,
    has_supplier_diversification: bool = False,
    has_dynamic_inventory: bool = False,
    has_flexible_transportation: bool = False,
    has_regional_flexibility: bool = False
) -> Dict[str, float]:
    """
    Run Monte Carlo simulation with given configuration and features.
    
    Args:
        config: Simulation configuration
        world: The simulation world
        has_supplier_diversification: Whether supplier diversification is enabled
        has_dynamic_inventory: Whether dynamic inventory management is enabled
        has_flexible_transportation: Whether flexible transportation routing is enabled
        has_regional_flexibility: Whether regional production flexibility is enabled
        
    Returns:
        Dict containing aggregated metrics from all simulation runs
    """
    # Set random seed for reproducibility
    random.seed(config['simulation']['seed'])
    np.random.seed(config['simulation']['seed'])
    
    # Update config based on features
    if has_supplier_diversification:
        config['supplier']['diversification_enabled'] = True
    if has_dynamic_inventory:
        config['inventory_management']['dynamic_enabled'] = True
    if has_flexible_transportation:
        config['logistics']['flexible_routing_enabled'] = True
    if has_regional_flexibility:
        config['production_facility']['regional_flexibility_enabled'] = True
    
    # Initialize metrics storage
    metrics_history = []
    
    # Run multiple iterations
    for _ in range(config['simulation']['monte_carlo_iterations']):
        # Reset world state for new iteration
        world.state = {}
        
        # Run simulation for specified time steps
        iteration_metrics = []
        for _ in range(config['simulation']['time_steps']):
            step_metrics = simulate_supply_chain_operation(world, config)
            iteration_metrics.append(step_metrics)
        
        # Calculate average metrics for this iteration
        avg_metrics = {
            key: np.mean([m[key] for m in iteration_metrics])
            for key in iteration_metrics[0].keys()
        }
        metrics_history.append(avg_metrics)
    
    # Calculate final aggregated metrics
    final_metrics = {
        'mean_completion_rate': np.mean([m['completion_rate'] for m in metrics_history]),
        'mean_on_time_delivery_rate': np.mean([m['on_time_delivery_rate'] for m in metrics_history]),
        'mean_average_delay': np.mean([m['average_delay'] for m in metrics_history]),
        'std_completion_rate': np.std([m['completion_rate'] for m in metrics_history]),
        'std_on_time_delivery_rate': np.std([m['on_time_delivery_rate'] for m in metrics_history]),
        'std_average_delay': np.std([m['average_delay'] for m in metrics_history]),
        'min_completion_rate': np.min([m['completion_rate'] for m in metrics_history]),
        'max_completion_rate': np.max([m['completion_rate'] for m in metrics_history]),
        'min_on_time_delivery_rate': np.min([m['on_time_delivery_rate'] for m in metrics_history]),
        'max_on_time_delivery_rate': np.max([m['on_time_delivery_rate'] for m in metrics_history]),
        'min_average_delay': np.min([m['average_delay'] for m in metrics_history]),
        'max_average_delay': np.max([m['average_delay'] for m in metrics_history]),
    }
    
    return final_metrics 