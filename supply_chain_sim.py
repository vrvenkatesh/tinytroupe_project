"""
Supply Chain Resilience Optimization Simulation - Main Entry Point

This module serves as the main entry point for running supply chain simulations
using TinyTroupe's agent-based simulation capabilities.
"""

from simulation import (
    DEFAULT_CONFIG,
    create_simulation_world,
    simulate_supply_chain_operation,
    run_monte_carlo_simulation
)

def main():
    """Main entry point for running supply chain simulations."""
    # Create simulation world with default configuration
    world = create_simulation_world(DEFAULT_CONFIG)
    
    # Run a single simulation step
    metrics = simulate_supply_chain_operation(world, DEFAULT_CONFIG)
    print("\nSingle step simulation metrics:")
    print(metrics)
    
    # Run Monte Carlo simulation with different feature combinations
    feature_combinations = [
        # (supplier_div, dynamic_inv, flexible_trans, regional_flex)
        (False, False, False, False),  # Baseline
        (True, False, False, False),   # Only supplier diversification
        (False, True, False, False),   # Only dynamic inventory
        (False, False, True, False),   # Only flexible transportation
        (False, False, False, True),   # Only regional flexibility
        (True, True, True, True),      # All features enabled
    ]
    
    print("\nMonte Carlo simulation results:")
    for features in feature_combinations:
        monte_carlo_metrics = run_monte_carlo_simulation(
            config=DEFAULT_CONFIG,
            world=world,
            has_supplier_diversification=features[0],
            has_dynamic_inventory=features[1],
            has_flexible_transportation=features[2],
            has_regional_flexibility=features[3]
        )
        
        print(f"\nFeatures enabled: {features}")
        print(f"Mean completion rate: {monte_carlo_metrics['mean_completion_rate']:.2f}")
        print(f"Mean on-time delivery rate: {monte_carlo_metrics['mean_on_time_delivery_rate']:.2f}")
        print(f"Mean average delay: {monte_carlo_metrics['mean_average_delay']:.2f} days")

if __name__ == "__main__":
    main() 