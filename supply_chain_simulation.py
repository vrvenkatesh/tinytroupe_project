#!/usr/bin/env python3
"""
Supply Chain Resilience Optimization Simulation - Main Execution Script

This script runs the complete Monte Carlo simulation for evaluating supply chain resilience
improvements using TinyTroupe's agent-based simulation capabilities.
"""

import os
import sys
import random
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime

# Add the tinytroupe directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'tinytroupe'))

from agent import TinyPerson
from environment.tiny_world import TinyWorld
from factory import TinyPersonFactory
from environment import logger
import config_init

# Import from the local package
from supply_chain import (
    DEFAULT_CONFIG,
    create_coo_agent,
    create_regional_manager_agent,
    create_supplier_agent,
    create_simulation_world,
    simulate_supply_chain_operation,
    export_comprehensive_results,
    Region,
)

def run_monte_carlo_simulation(
    config: Dict[str, Any],
    world: TinyWorld,
    has_supplier_diversification: bool = False,
    has_dynamic_inventory: bool = False,
    has_flexible_transportation: bool = False,
    has_regional_flexibility: bool = False
) -> Dict[str, float]:
    """Run Monte Carlo simulation for supply chain operations."""
    results = []
    random.seed(config['simulation']['seed'])
    
    # Initialize tracking for daily metrics across all iterations
    daily_metrics = {
        'resilience_score': [],
        'completion_rate': [],
        'on_time_delivery_rate': [],
        'risk_level': [],
        'service_level': []
    }
    
    # Track order lifecycle and agent interactions across iterations
    daily_order_tracking = []
    daily_interaction_tracking = []
    
    for iteration in range(config['simulation']['monte_carlo_iterations']):
        # Create unique simulation ID
        simulation_id = f"{world.name}_iter_{iteration}"
        iteration_results = []
        
        # Reset world state for this iteration
        world.state = {
            'risk_exposure': 0.5,
            'cost_pressure': 0.5,
            'demand_volatility': 0.5,
            'supply_risk': 0.5,
            'reliability_requirement': 0.5,
            'flexibility_requirement': 0.5,
            'active_orders': [],
            'completed_orders': [],
            'order_lifecycle': {},  # Initialize as dictionary
            'regional_metrics': {
                region.value: {
                    'risk': 0.5,
                    'cost': 0.5,
                    'demand': 0.5,
                    'supply_risk': 0.5,
                    'infrastructure': 0.7,
                    'congestion': 0.3,
                    'efficiency': 0.8,
                    'flexibility': 0.7,
                    'quality': 0.8
                } for region in Region
            }
        }
        world.current_datetime = datetime.now()  # Reset datetime for this iteration
        
        # Reset agent interactions for this iteration
        for agent in world.agents:
            agent.interactions = []
        
        # Configure supply chain capabilities and adjust initial metrics
        base_metrics = config['coo'].get('initial_metrics', {})
        feature_multiplier = 1.0
        
        if has_supplier_diversification:
            feature_multiplier *= 1.2
            world.state['supply_risk'] *= 0.8
            
        if has_dynamic_inventory:
            feature_multiplier *= 1.15
            world.state['cost_pressure'] *= 0.85
            
        if has_flexible_transportation:
            feature_multiplier *= 1.1
            world.state['reliability_requirement'] *= 0.9
            
        if has_regional_flexibility:
            feature_multiplier *= 1.25
            world.state['flexibility_requirement'] *= 0.75
        
        # Run simulation steps
        for step in range(config['simulation']['time_steps']):
            world.current_time = step
            step_results = simulate_supply_chain_operation(
                world=world,
                config=config
            )
            
            # Store a snapshot of all orders and interactions for this day
            active_orders = world.state.get('active_orders', [])
            completed_orders = world.state.get('completed_orders', [])
            all_orders = active_orders + completed_orders
            
            # Create order lifecycle snapshot
            order_snapshot = []
            for order in all_orders:
                order_data = {
                    'order_id': order.id,
                    'status': order.status.value,
                    'current_location': order.current_location.value,
                    'source_region': order.source_region.value,
                    'destination_region': order.destination_region.value,
                    'creation_time': order.creation_time,
                    'expected_delivery_time': order.expected_delivery_time,
                    'actual_delivery_time': order.actual_delivery_time,
                    'production_time': order.production_time,
                    'transit_time': order.transit_time,
                    'delay_time': order.delay_time,
                    'transportation_mode': order.transportation_mode.value if order.transportation_mode else None,
                    'current_handler': order.current_handler,
                    'simulation_day': step
                }
                order_snapshot.append(order_data)
            daily_order_tracking.append(order_snapshot)
            
            # Collect all agent interactions
            interaction_snapshot = []
            for agent in world.agents:
                for interaction in getattr(agent, 'interactions', []):
                    if isinstance(interaction, dict):  # Already a dict
                        interaction_data = interaction.copy()
                    else:  # Custom interaction object
                        interaction_data = {
                            'agent_id': agent.id,
                            'agent_type': agent.role,
                            'interaction_type': getattr(interaction, 'type', 'unknown'),
                            'timestamp': getattr(interaction, 'timestamp', datetime.now()),
                            'target_agent': getattr(interaction, 'target_agent', 'unknown'),
                            'order_id': getattr(interaction, 'order_id', 'unknown'),
                            'status': getattr(interaction, 'status', 'unknown'),
                            'success': getattr(interaction, 'success', False),
                            'message': getattr(interaction, 'message', ''),
                            'simulation_day': step
                        }
                    interaction_snapshot.append(interaction_data)
            daily_interaction_tracking.append(interaction_snapshot)
            
            # Calculate total orders for this step
            step_results['total_orders'] = len(all_orders)
            
            # Remove current_datetime from step_results before appending
            if 'current_datetime' in step_results:
                del step_results['current_datetime']
            iteration_results.append(step_results)
            
            # Track daily metrics
            for metric in daily_metrics:
                if metric in step_results:
                    daily_metrics[metric].append(step_results[metric])
                else:
                    daily_metrics[metric].append(0)  # Default to 0 if metric not present
        
        # Aggregate results for this iteration
        iteration_aggregated = {
            'mean_resilience_score': np.mean([r.get('resilience_score', 0) for r in iteration_results]),
            'std_resilience_score': np.std([r.get('resilience_score', 0) for r in iteration_results]),
            'min_resilience_score': min([r.get('resilience_score', 0) for r in iteration_results]),
            'max_resilience_score': max([r.get('resilience_score', 0) for r in iteration_results]),
            'mean_completion_rate': np.mean([r.get('completion_rate', 0) for r in iteration_results]),
            'std_completion_rate': np.std([r.get('completion_rate', 0) for r in iteration_results]),
            'min_completion_rate': min([r.get('completion_rate', 0) for r in iteration_results]),
            'max_completion_rate': max([r.get('completion_rate', 0) for r in iteration_results]),
            'mean_on_time_delivery_rate': np.mean([r.get('on_time_delivery_rate', 0) for r in iteration_results]),
            'std_on_time_delivery_rate': np.std([r.get('on_time_delivery_rate', 0) for r in iteration_results]),
            'min_on_time_delivery_rate': min([r.get('on_time_delivery_rate', 0) for r in iteration_results]),
            'max_on_time_delivery_rate': max([r.get('max_on_time_delivery_rate', 0) for r in iteration_results]),
            'mean_risk_level': np.mean([r.get('risk_level', 0) for r in iteration_results]),
            'std_risk_level': np.std([r.get('risk_level', 0) for r in iteration_results]),
            'min_risk_level': min([r.get('min_risk_level', 0) for r in iteration_results]),
            'max_risk_level': max([r.get('max_risk_level', 0) for r in iteration_results]),
            'feature_multiplier': feature_multiplier
        }
        
        results.append(iteration_aggregated)
    
    # Calculate final aggregated metrics across all iterations
    final_metrics = {
        'mean_resilience_score': np.mean([r['mean_resilience_score'] for r in results]),
        'std_resilience_score': np.mean([r['std_resilience_score'] for r in results]),
        'min_resilience_score': min([r['min_resilience_score'] for r in results]),
        'max_resilience_score': max([r['max_resilience_score'] for r in results]),
        'mean_completion_rate': np.mean([r['mean_completion_rate'] for r in results]),
        'std_completion_rate': np.mean([r['std_completion_rate'] for r in results]),
        'min_completion_rate': min([r['min_completion_rate'] for r in results]),
        'max_completion_rate': max([r['max_completion_rate'] for r in results]),
        'mean_on_time_delivery_rate': np.mean([r['mean_on_time_delivery_rate'] for r in results]),
        'std_on_time_delivery_rate': np.mean([r['std_on_time_delivery_rate'] for r in results]),
        'min_on_time_delivery_rate': min([r['min_on_time_delivery_rate'] for r in results]),
        'max_on_time_delivery_rate': max([r['max_on_time_delivery_rate'] for r in results]),
        'mean_risk_level': np.mean([r['mean_risk_level'] for r in results]),
        'std_risk_level': np.mean([r['std_risk_level'] for r in results]),
        'min_risk_level': min([r['min_risk_level'] for r in results]),
        'max_risk_level': max([r['max_risk_level'] for r in results]),
        'feature_multiplier': np.mean([r['feature_multiplier'] for r in results])
    }
    
    # Store the final order lifecycle and agent interactions in the world state
    world.state['order_lifecycle'] = daily_order_tracking
    world.state['agent_interactions'] = daily_interaction_tracking
    
    return final_metrics

def visualize_results(
    baseline_results: Dict[str, Any],
    supplier_diversification_results: Dict[str, Any],
    dynamic_inventory_results: Dict[str, Any],
    flexible_transportation_results: Dict[str, Any],
    regional_flexibility_results: Dict[str, Any],
    combined_results: Dict[str, Any]
):
    """Generate visualizations comparing different resilience strategies."""
    metrics = ['resilience_score', 'service_level', 'total_cost', 'risk_exposure']
    scenarios = [
        'Baseline',
        'Supplier Diversification',
        'Dynamic Inventory',
        'Flexible Transportation',
        'Regional Flexibility',
        'Combined'
    ]
    
    results = [
        baseline_results,
        supplier_diversification_results,
        dynamic_inventory_results,
        flexible_transportation_results,
        regional_flexibility_results,
        combined_results
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Supply Chain Resilience Strategy Comparison')
    
    for idx, metric in enumerate(metrics):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        means = [r[metric]['mean'] for r in results]
        stds = [r[metric]['std'] for r in results]
        
        ax.bar(scenarios, means, yerr=stds, capsize=5)
        ax.set_title(metric.replace('_', ' ').title())
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('supply_chain_simulation_results.png')

def main():
    """Main execution function for the supply chain simulation."""
    # Load configuration
    config = DEFAULT_CONFIG
    
    # Create simulation world
    world = create_simulation_world(config)
    
    # Run baseline simulation
    print("Running baseline simulation...")
    baseline_results = run_monte_carlo_simulation(config, world)
    
    # Run simulations with individual improvements
    print("Running simulation with supplier diversification...")
    supplier_diversification_results = run_monte_carlo_simulation(
        config, world, has_supplier_diversification=True
    )
    
    print("Running simulation with dynamic inventory...")
    dynamic_inventory_results = run_monte_carlo_simulation(
        config, world, has_dynamic_inventory=True
    )
    
    print("Running simulation with flexible transportation...")
    flexible_transportation_results = run_monte_carlo_simulation(
        config, world, has_flexible_transportation=True
    )
    
    print("Running simulation with regional flexibility...")
    regional_flexibility_results = run_monte_carlo_simulation(
        config, world, has_regional_flexibility=True
    )
    
    # Run simulation with all improvements combined
    print("Running simulation with all improvements combined...")
    combined_results = run_monte_carlo_simulation(
        config, world,
        has_supplier_diversification=True,
        has_dynamic_inventory=True,
        has_flexible_transportation=True,
        has_regional_flexibility=True
    )
    
    # Generate visualizations
    print("Generating visualizations...")
    visualize_results(
        baseline_results,
        supplier_diversification_results,
        dynamic_inventory_results,
        flexible_transportation_results,
        regional_flexibility_results,
        combined_results
    )
    
    # Export results
    print("Exporting results...")
    export_comprehensive_results(
        baseline_results,
        supplier_diversification_results,
        dynamic_inventory_results,
        flexible_transportation_results,
        regional_flexibility_results,
        combined_results,
        'supply_chain_simulation_results.csv'
    )

if __name__ == "__main__":
    main() 