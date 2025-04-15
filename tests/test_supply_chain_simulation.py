"""
Test suite for supply chain simulation with simplified configuration for faster execution.
"""

import unittest
import os
import pandas as pd
import numpy as np
from typing import Dict, Any
import uuid
import random
import csv
import time

from supply_chain_simulation import (
    run_monte_carlo_simulation,
    visualize_results,
    DEFAULT_CONFIG,
    create_simulation_world
)

class TestSupplyChainSimulation(unittest.TestCase):
    """Test cases for supply chain simulation."""

    def setUp(self):
        """Set up test fixtures with minimal configuration."""
        print("\n=== Setting up simplified test environment ===")
        self.config = DEFAULT_CONFIG.copy()
        
        # Set random seed for reproducibility
        random.seed(42)
        self.simulation_id = str(uuid.uuid4())[:8]
        
        # Set up output files
        self.results_dir = "test_results"
        os.makedirs(self.results_dir, exist_ok=True)
        self.csv_file = os.path.join(self.results_dir, f"simulation_results_{self.simulation_id}.csv")
        self.md_file = os.path.join(self.results_dir, f"simulation_results_{self.simulation_id}.md")
        
        # Reduce simulation complexity
        self.config['simulation'].update({
            'monte_carlo_iterations': 5,
            'suppliers_per_region': 2,
            'time_steps': 5,
            'regions': ['NORTH_AMERICA', 'EUROPE'],
            'seed': 42
        })
        
        # Simplify agent configurations for faster processing
        base_agent_config = {
            'max_actions': 2,
            'verbosity': 'high',
            'max_tokens': 50,
            'temperature': 0.7,
            'initial_metrics': {
                'resilience_score': 0.7,
                'recovery_time': 0.6,
                'service_level': 0.8,
                'total_cost': 0.4,
                'inventory_cost': 0.5,
                'transportation_cost': 0.3,
                'risk_exposure': 0.4,
                'supplier_risk': 0.3,
                'transportation_risk': 0.4,
                'lead_time': 0.5,
                'flexibility_score': 0.6,
                'quality_score': 0.8
            }
        }
        
        self.config['coo'].update(base_agent_config)
        self.config['regional_manager'].update(base_agent_config)
        self.config['supplier'].update(base_agent_config)
        
        # Create test world with minimal regions
        self.world = create_simulation_world(self.config)
        self.world.name = f"FastTestWorld_{self.simulation_id}"
        self.output_file = f"test_simulation_results_{self.simulation_id}.csv"
        self.output_plot = f"test_simulation_results_{self.simulation_id}.png"
        
        print(f"Created test world: {self.world.name}")
        print("Using simplified configuration:")
        print(f"- Monte Carlo iterations: {self.config['simulation']['monte_carlo_iterations']}")
        print(f"- Suppliers per region: {self.config['simulation']['suppliers_per_region']}")
        print(f"- Time steps: {self.config['simulation']['time_steps']}")
        print(f"- Regions: {self.config['simulation']['regions']}")

    def tearDown(self):
        """Clean up test artifacts and agents."""
        print("\n=== Cleaning up test artifacts and agents ===")
        
        # Clean up files
        for file_path in [self.output_file, self.output_plot]:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Removed file: {file_path}")
        
        # Clean up world and agents
        if hasattr(self, 'world'):
            if self.world.name in self.world.all_environments:
                print(f"Removing world: {self.world.name}")
                # Remove all agents from the world first
                for agent in self.world.agents[:]:
                    self.world.remove_agent(agent)
                del self.world.all_environments[self.world.name]
        
        # Force cleanup of any remaining agents
        from tinytroupe.agent.tiny_person import TinyPerson
        if hasattr(TinyPerson, 'all_agents'):
            agent_count = len(TinyPerson.all_agents)
            TinyPerson.all_agents.clear()
            print(f"Cleaned up {agent_count} agents")
        
        print("Cleanup complete")

    def _save_results_to_csv(self, baseline: Dict[str, Any], improved: Dict[str, Any]):
        """Save simulation results to CSV file."""
        metric_groups = {
            'Core Metrics': ['resilience_score', 'recovery_time', 'service_level'],
            'Cost Metrics': ['total_cost', 'inventory_cost', 'transportation_cost'],
            'Risk Metrics': ['risk_exposure', 'supplier_risk', 'transportation_risk'],
            'Performance Metrics': ['lead_time', 'flexibility_score', 'quality_score']
        }
        
        rows = []
        # Header row
        rows.append(['Metric Group', 'Metric', 
                    'Baseline Mean', 'Baseline Std', 'Baseline Min', 'Baseline Max',
                    'Improved Mean', 'Improved Std', 'Improved Min', 'Improved Max',
                    'Absolute Change', 'Relative Change (%)'])
        
        # Data rows
        for group_name, metrics in metric_groups.items():
            for metric in metrics:
                if metric in baseline and metric in improved:
                    b_values = baseline[metric]
                    i_values = improved[metric]
                    diff = i_values['mean'] - b_values['mean']
                    rel_change = (diff / b_values['mean']) * 100 if b_values['mean'] != 0 else float('inf')
                    
                    rows.append([
                        group_name,
                        metric.replace('_', ' ').title(),
                        f"{b_values['mean']:.3f}",
                        f"{b_values['std']:.3f}",
                        f"{b_values['min']:.3f}",
                        f"{b_values['max']:.3f}",
                        f"{i_values['mean']:.3f}",
                        f"{i_values['std']:.3f}",
                        f"{i_values['min']:.3f}",
                        f"{i_values['max']:.3f}",
                        f"{diff:+.3f}",
                        f"{rel_change:+.1f}"
                    ])
        
        # Write to CSV
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        print(f"\nResults saved to CSV: {self.csv_file}")

    def _save_results_to_markdown(self, baseline: Dict[str, Any], improved: Dict[str, Any]):
        """Save simulation results to Markdown file."""
        metric_groups = {
            'Core Metrics': ['resilience_score', 'recovery_time', 'service_level'],
            'Cost Metrics': ['total_cost', 'inventory_cost', 'transportation_cost'],
            'Risk Metrics': ['risk_exposure', 'supplier_risk', 'transportation_risk'],
            'Performance Metrics': ['lead_time', 'flexibility_score', 'quality_score']
        }
        
        with open(self.md_file, 'w') as f:
            # Write header
            f.write(f"# Supply Chain Simulation Results\n\n")
            f.write(f"Simulation ID: {self.simulation_id}  \n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}  \n\n")
            
            # Write configuration
            f.write("## Configuration\n\n")
            f.write(f"- Monte Carlo Iterations: {self.config['simulation']['monte_carlo_iterations']}\n")
            f.write(f"- Suppliers per Region: {self.config['simulation']['suppliers_per_region']}\n")
            f.write(f"- Time Steps: {self.config['simulation']['time_steps']}\n")
            f.write(f"- Regions: {self.config['simulation']['regions']}\n\n")
            
            # Write results for each metric group
            f.write("## Results\n\n")
            for group_name, metrics in metric_groups.items():
                f.write(f"### {group_name}\n\n")
                
                # Table header
                f.write("| Metric | Baseline (μ ± σ) | Improved (μ ± σ) | Change |\n")
                f.write("|--------|-----------------|-----------------|--------|\n")
                
                for metric in metrics:
                    if metric in baseline and metric in improved:
                        b_values = baseline[metric]
                        i_values = improved[metric]
                        diff = i_values['mean'] - b_values['mean']
                        rel_change = (diff / b_values['mean']) * 100 if b_values['mean'] != 0 else float('inf')
                        
                        metric_name = metric.replace('_', ' ').title()
                        baseline_str = f"{b_values['mean']:.3f} ± {b_values['std']:.3f}"
                        improved_str = f"{i_values['mean']:.3f} ± {i_values['std']:.3f}"
                        change_str = f"{diff:+.3f} ({rel_change:+.1f}%)"
                        
                        f.write(f"| {metric_name} | {baseline_str} | {improved_str} | {change_str} |\n")
                
                f.write("\n")
                
            # Write detailed statistics
            f.write("## Detailed Statistics\n\n")
            for group_name, metrics in metric_groups.items():
                f.write(f"### {group_name}\n\n")
                for metric in metrics:
                    if metric in baseline and metric in improved:
                        f.write(f"#### {metric.replace('_', ' ').title()}\n\n")
                        f.write("| Scenario | Mean | Std Dev | Min | Max |\n")
                        f.write("|----------|------|---------|-----|-----|\n")
                        
                        b_values = baseline[metric]
                        i_values = improved[metric]
                        
                        f.write(f"| Baseline | {b_values['mean']:.3f} | {b_values['std']:.3f} | {b_values['min']:.3f} | {b_values['max']:.3f} |\n")
                        f.write(f"| Improved | {i_values['mean']:.3f} | {i_values['std']:.3f} | {i_values['min']:.3f} | {i_values['max']:.3f} |\n\n")
        
        print(f"\nResults saved to Markdown: {self.md_file}")

    def test_quick_simulation(self):
        """Run a quick simulation with minimal configuration."""
        print("\n=== Running Quick Simulation Test ===")
        
        print("\n1. Running baseline scenario...")
        baseline_results = run_monte_carlo_simulation(
            config=self.config,
            world=self.world,
            has_supplier_diversification=False,
            has_dynamic_inventory=False,
            has_flexible_transportation=False,
            has_regional_flexibility=False
        )
        print("\nBaseline Results:")
        self._print_results(baseline_results)
        
        # Clean up agents between runs
        print("\nCleaning up agents before improved scenario...")
        from tinytroupe.agent.tiny_person import TinyPerson
        if hasattr(TinyPerson, 'all_agents'):
            agent_count = len(TinyPerson.all_agents)
            TinyPerson.all_agents.clear()
            print(f"Cleaned up {agent_count} agents")
        
        # Also clean up world agents
        for agent in self.world.agents[:]:
            self.world.remove_agent(agent)
        print("Cleared world agents")
        
        print("\n2. Running improved scenario...")
        improved_results = run_monte_carlo_simulation(
            config=self.config,
            world=self.world,
            has_supplier_diversification=True,
            has_dynamic_inventory=True,
            has_flexible_transportation=True,
            has_regional_flexibility=True
        )
        print("\nImproved Results:")
        self._print_results(improved_results)
        
        # Basic assertions to verify simulation output
        self._verify_results(baseline_results)
        self._verify_results(improved_results)
        
        # Compare improvements
        self._compare_results(baseline_results, improved_results)
        
        # Save results to files
        self._save_results_to_csv(baseline_results, improved_results)
        self._save_results_to_markdown(baseline_results, improved_results)
        
        print("\nQuick simulation test completed successfully!")

    def _print_results(self, results: Dict[str, Any]):
        """Helper method to print formatted results."""
        print("\n" + "="*80)
        print("SIMULATION RESULTS")
        print("="*80)
        
        # Print metrics in groups
        metric_groups = {
            'Core Metrics': ['resilience_score', 'recovery_time', 'service_level'],
            'Cost Metrics': ['total_cost', 'inventory_cost', 'transportation_cost'],
            'Risk Metrics': ['risk_exposure', 'supplier_risk', 'transportation_risk'],
            'Performance Metrics': ['lead_time', 'flexibility_score', 'quality_score']
        }
        
        for group_name, metrics in metric_groups.items():
            print(f"\n{group_name}:")
            print("-" * 40)
            for metric in metrics:
                if metric in results:
                    values = results[metric]
                    print(f"  {metric.replace('_', ' ').title()}:")
                    print(f"    Mean: {values['mean']:.3f} ± {values['std']:.3f}")
                    print(f"    Range: [{values['min']:.3f} - {values['max']:.3f}]")

    def _verify_results(self, results: Dict[str, Any]):
        """Helper method to verify result structure and values."""
        for metric, values in results.items():
            self.assertIsInstance(values, dict)
            self.assertTrue(0 <= values['mean'] <= 1, f"{metric} mean out of range")
            self.assertTrue(0 <= values['min'] <= values['max'] <= 1, f"{metric} min/max out of range")
            # Ensure we're getting non-zero results
            self.assertNotEqual(values['mean'], 0, f"{metric} mean should not be zero")

    def _compare_results(self, baseline: Dict[str, Any], improved: Dict[str, Any]):
        """Helper method to compare baseline and improved results."""
        print("\n" + "="*80)
        print("IMPROVEMENTS FROM BASELINE")
        print("="*80)
        
        metric_groups = {
            'Core Metrics': ['resilience_score', 'recovery_time', 'service_level'],
            'Cost Metrics': ['total_cost', 'inventory_cost', 'transportation_cost'],
            'Risk Metrics': ['risk_exposure', 'supplier_risk', 'transportation_risk'],
            'Performance Metrics': ['lead_time', 'flexibility_score', 'quality_score']
        }
        
        for group_name, metrics in metric_groups.items():
            print(f"\n{group_name}:")
            print("-" * 40)
            for metric in metrics:
                if metric in baseline and metric in improved:
                    diff = improved[metric]['mean'] - baseline[metric]['mean']
                    rel_change = (diff / baseline[metric]['mean']) * 100 if baseline[metric]['mean'] != 0 else float('inf')
                    print(f"  {metric.replace('_', ' ').title()}:")
                    print(f"    Absolute change: {diff:+.3f}")
                    print(f"    Relative change: {rel_change:+.1f}%")

if __name__ == '__main__':
    unittest.main(verbosity=2) 