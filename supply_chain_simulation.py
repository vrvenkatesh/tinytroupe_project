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
    
    for iteration in range(config['simulation']['monte_carlo_iterations']):
        # Create unique simulation ID
        simulation_id = f"{world.name}_iter_{iteration}"
        
        # Create agents with updated configurations
        coo_config = config['coo'].copy()
        regional_config = config['regional_manager'].copy()
        supplier_config = config['supplier'].copy()
        logistics_config = config['logistics'].copy()
        production_config = config['production_facility'].copy()
        
        # Configure supply chain capabilities and adjust initial metrics
        base_metrics = coo_config.get('initial_metrics', {})
        improvement_factor = 1.0
        
        if has_supplier_diversification:
            supplier_config['diversification_enabled'] = True
            improvement_factor += 0.1
            base_metrics['supplier_risk'] = max(0, base_metrics.get('supplier_risk', 0.5) * 0.8)
            
        if has_dynamic_inventory:
            inventory_config = config['inventory_management'].copy()
            inventory_config['dynamic_enabled'] = True
            improvement_factor += 0.1
            base_metrics['inventory_cost'] = max(0, base_metrics.get('inventory_cost', 0.5) * 0.85)
            
        if has_flexible_transportation:
            logistics_config['flexible_routing_enabled'] = True
            improvement_factor += 0.1
            base_metrics['transportation_risk'] = max(0, base_metrics.get('transportation_risk', 0.5) * 0.8)
            
        if has_regional_flexibility:
            production_config['regional_flexibility_enabled'] = True
            improvement_factor += 0.1
            base_metrics['flexibility_score'] = min(1.0, base_metrics.get('flexibility_score', 0.5) * 1.2)
        
        # Update base metrics with improvements
        for metric in base_metrics:
            if metric not in ['supplier_risk', 'inventory_cost', 'transportation_risk', 'flexibility_score']:
                base_metrics[metric] = min(1.0, base_metrics[metric] * improvement_factor)
        
        # Create COO agent with unique name
        coo = create_coo_agent(
            f"COO_{simulation_id}_{iteration}",
            {**coo_config, 'initial_metrics': base_metrics.copy()},
            simulation_id
        )
        
        # Create regional managers with unique names
        regional_managers = {
            region: create_regional_manager_agent(
                f"Manager_{region.name}_{simulation_id}_{iteration}",
                {**regional_config, 'initial_metrics': base_metrics.copy()},
                simulation_id
            )
            for region in world.regions
        }
        
        # Create suppliers with unique names
        suppliers = {
            region: [
                create_supplier_agent(
                    f"Supplier_{region.name}_{i}_{simulation_id}_{iteration}",
                    {**supplier_config, 'initial_metrics': base_metrics.copy()},
                    simulation_id
                )
                for i in range(config['simulation']['suppliers_per_region'])
            ]
            for region in world.regions
        }
        
        # Add all agents to the world
        world.add_agent(coo)
        for manager in regional_managers.values():
            world.add_agent(manager)
        for region_suppliers in suppliers.values():
            for supplier in region_suppliers:
                world.add_agent(supplier)
        
        # Run simulation for multiple time steps
        iteration_results = []
        daily_metrics = {
            'active_orders': [],
            'completed_orders': [],
            'delayed_orders': [],
            'service_level': [],
            'resilience_score': [],
            'lead_time': []
        }
        
        daily_order_status = []
        
        for _ in range(config['simulation']['time_steps']):
            step_results = simulate_supply_chain_operation(
                world=world,
                config=config
            )
            # Remove current_datetime from step_results before appending
            if 'current_datetime' in step_results:
                del step_results['current_datetime']
            iteration_results.append(step_results)
            
            # Track daily metrics
            for metric in daily_metrics:
                daily_metrics[metric].append(step_results[metric])
            
            # Track daily order status separately
            if 'order_status' in step_results:
                daily_order_status.append(step_results['order_status'])
        
        # Aggregate results for this iteration
        iteration_aggregated = {}
        for metric in iteration_results[0].keys():
            if metric == 'order_status':
                continue  # Skip order_status as it's handled separately
            
            values = [r[metric] for r in iteration_results]
            if metric == 'delayed_orders':
                # For delayed_orders, take the maximum value across time steps
                iteration_aggregated[metric] = {
                    'mean': float(max(values)),
                    'std': float(np.std(values)),
                    'min': float(min(values)),
                    'max': float(max(values)),
                    'daily': values  # Add daily values
                }
            else:
                # For other metrics, take the mean across time steps
                iteration_aggregated[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(max(values)),
                    'daily': values  # Add daily values
                }
        
        # Add order status to iteration results
        iteration_aggregated['order_status'] = {
            'daily': daily_order_status
        }
        
        results.append(iteration_aggregated)
        
        # Clean up agents after each iteration
        for agent in world.agents[:]:
            world.remove_agent(agent)
    
    # Aggregate results across all iterations
    aggregated_results = {}
    for metric in results[0].keys():
        if metric == 'order_status':
            # For order status, just take the first iteration's values for simplicity
            aggregated_results[metric] = results[0][metric]
            continue
        
        values = [r[metric]['mean'] for r in results]  # Use mean from each iteration
        daily_values = [r[metric]['daily'] for r in results]  # Get daily values from all iterations
        
        # Calculate mean daily values across iterations
        mean_daily = []
        for day in range(len(daily_values[0])):
            day_values = [iteration[day] for iteration in daily_values]
            mean_daily.append(float(np.mean(day_values)))
        
        aggregated_results[metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'daily': mean_daily  # Add mean daily values
        }
    
    return aggregated_results

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