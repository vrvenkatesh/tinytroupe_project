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
from datetime import datetime, timedelta
import json
from pathlib import Path

from supply_chain_simulation import (
    run_monte_carlo_simulation,
    DEFAULT_CONFIG,
    create_simulation_world,
    create_coo_agent,
    create_regional_manager_agent,
    create_supplier_agent,
    create_logistics_provider_agent,
    create_production_facility_agent,
    simulate_supply_chain_operation,
    Region,
    OrderStatus,
    TransportationMode
)

from tinytroupe.extraction import ResultsExtractor, ResultsReducer
from tinytroupe.environment.tiny_world import TinyWorld
from tinytroupe.agent.tiny_person import TinyPerson
from tests.test_helpers import TestArtifactGenerator

class TestSupplyChainSimulation(unittest.TestCase):
    """Test cases for supply chain simulation."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that should be created only once."""
        # Ensure parent test_results directory exists
        if not os.path.exists("test_results"):
            os.makedirs("test_results")

    def clean_registries(self):
        """Clean up all agent and world registries."""
        print("\n=== Cleaning up registries ===")
        
        # First, remove all agents from all worlds
        if hasattr(TinyWorld, 'all_environments'):
            for world_name, world in list(TinyWorld.all_environments.items()):
                # Remove all agents from this world
                for agent_name in list(world.name_to_agent.keys()):
                    try:
                        agent = world.name_to_agent[agent_name]
                        world.remove_agent(agent)
                        print(f"Removed agent {agent_name} from world {world_name}")
                    except Exception as e:
                        print(f"Warning: Failed to remove agent {agent_name} from world {world_name}: {str(e)}")
                
                # Remove world from registry
                try:
                    del TinyWorld.all_environments[world_name]
                    print(f"Removed world {world_name} from environments registry")
                except Exception as e:
                    print(f"Warning: Failed to remove world {world_name}: {str(e)}")
        
        # Then clean up the global agent registry
        if hasattr(TinyPerson, 'all_agents'):
            for agent_name in list(TinyPerson.all_agents.keys()):
                try:
                    del TinyPerson.all_agents[agent_name]
                    print(f"Removed agent {agent_name} from global registry")
                except Exception as e:
                    print(f"Warning: Failed to remove agent {agent_name} from global registry: {str(e)}")
        
        # Final verification
        if hasattr(TinyPerson, 'all_agents'):
            assert len(TinyPerson.all_agents) == 0, "Global agent registry not empty"
        if hasattr(TinyWorld, 'all_environments'):
            assert len(TinyWorld.all_environments) == 0, "World registry not empty"
        
        print("=== Registry cleanup complete ===")
        
        # Reset instance variables
        self.world = None
        self.coo = None
        self.regional_managers = {}
        self.suppliers = {}
        self.production_facilities = {}
        self.logistics_providers = []

    def setUp(self):
        """Set up test fixtures."""
        # Clean all registries before setting up new test
        self.clean_registries()
        
        self.simulation_id = str(uuid.uuid4())[:8]
        self.config = DEFAULT_CONFIG.copy()
        
        # Initialize test artifact generator with simulation-specific directory
        self.artifact_dir = os.path.join("test_results", f"simulation_{self.simulation_id}")
        os.makedirs(self.artifact_dir, exist_ok=True)
        self.artifact_generator = TestArtifactGenerator(
            simulation_id=self.simulation_id,
            output_dir=self.artifact_dir
        )
        
        # Simplify configuration for testing
        self.config['simulation'].update({
            'monte_carlo_iterations': 2,
            'suppliers_per_region': 2,
            'time_steps': 10,
            'base_demand': 5,
            'regions': ['NORTH_AMERICA', 'EUROPE']
        })
        
        self.config['production_facility'].update({
            'base_production_time': 2
        })
        
        # Create test world with unique name
        self.world = create_simulation_world(self.config)
        
        # Create agents
        self.coo = create_coo_agent(
            f"COO_{self.simulation_id}",
            self.config['coo'],
            self.world.name
        )
        
        # Create regional managers
        self.regional_managers = {
            region: create_regional_manager_agent(
                f"RegionalManager_{region.value}_{self.simulation_id}",
                {**self.config['regional_manager'], 'region': region},
                self.world.name
            ) for region in [Region.NORTH_AMERICA, Region.EUROPE]
        }
        
        # Create suppliers
        self.suppliers = {
            region: [
                create_supplier_agent(
                    f"Supplier_{region.value}_{i}_{self.simulation_id}",
                    {**self.config['supplier'], 'region': region},
                    self.world.name
                ) for i in range(self.config['simulation']['suppliers_per_region'])
            ] for region in [Region.NORTH_AMERICA, Region.EUROPE]
        }
        
        # Create production facilities
        self.production_facilities = {
            region: create_production_facility_agent(
                f"ProductionFacility_{region.value}_{self.simulation_id}",
                {**self.config['production_facility'], 'region': region},
                self.world.name
            ) for region in [Region.NORTH_AMERICA, Region.EUROPE]
        }
        
        # Create logistics providers
        self.logistics_providers = []
        for source_region in [Region.NORTH_AMERICA, Region.EUROPE]:
            for dest_region in [Region.NORTH_AMERICA, Region.EUROPE]:
                if source_region != dest_region:  # Only create providers for different regions
                    provider = create_logistics_provider_agent(
                        source_region=source_region,
                        dest_region=dest_region,
                        config=self.config['logistics']
                    )
                    self.logistics_providers.append(provider)
        
        # Add agents to world
        self.world.add_agent(self.coo)
        for manager in self.regional_managers.values():
            self.world.add_agent(manager)
        for region_suppliers in self.suppliers.values():
            for supplier in region_suppliers:
                self.world.add_agent(supplier)
        for facility in self.production_facilities.values():
            self.world.add_agent(facility)
        for provider in self.logistics_providers:
            self.world.add_agent(provider)
        
        # Store agents in world state for simulation access
        self.world.state.update({
            'coo_agent': self.coo,
            'regional_managers': list(self.regional_managers.values()),
            'suppliers': [s for suppliers in self.suppliers.values() for s in suppliers],
            'production_facilities': list(self.production_facilities.values()),
            'logistics_providers': self.logistics_providers,
            'active_orders': [],
            'completed_orders': [],
            'order_lifecycle': {},
        })
        
        print("Using simplified configuration:")
        print(f"- Monte Carlo iterations: {self.config['simulation']['monte_carlo_iterations']}")
        print(f"- Suppliers per region: {self.config['simulation']['suppliers_per_region']}")
        print(f"- Time steps: {self.config['simulation']['time_steps']}")
        print(f"- Regions: {self.config['simulation']['regions']}")
        print(f"- Total agents created: {len(self.world.agents)}")
        
        # Verify agent registration
        print("\nVerifying agent registration:")
        print(f"- Agents in world: {len(self.world.agents)}")
        print(f"- Agents in name_to_agent: {len(self.world.name_to_agent)}")
        print(f"- Agents in global registry: {len(TinyPerson.all_agents)}")

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up test artifacts
        test_files = [
            f"test_results/baseline_metrics_{self.simulation_id}.csv",
            f"test_results/improved_metrics_{self.simulation_id}.csv",
            f"test_results/comparison_{self.simulation_id}.csv"
        ]
        
        for file in test_files:
            if os.path.exists(file):
                try:
                    os.remove(file)
                    print(f"Removed test file: {file}")
                except Exception as e:
                    print(f"Warning: Could not remove {file}: {e}")
        
        # Clean up all registries
        self.clean_registries()
        
        # Clear any plots
        if self.output_plot:
            self.output_plot.close()
            self.output_plot = None

    def _dict_to_csv(self, data: Dict[str, Any], output_file: str) -> None:
        """Save dictionary data to CSV file."""
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write Core Metrics section
            writer.writerow(['Core Metrics'])
            writer.writerow(['Metric', 'Mean', 'Std Dev', 'Min', 'Max'])
            core_metrics = [
                'resilience_score', 'recovery_time', 'service_level',
                'total_cost', 'inventory_cost', 'transportation_cost',
                'risk_exposure', 'supplier_risk', 'transportation_risk',
                'lead_time', 'flexibility_score', 'quality_score',
                'total_interactions', 'unique_interacting_agents', 'total_orders'
            ]
            for metric in core_metrics:
                if metric in data:
                    metric_data = data[metric]
                    if isinstance(metric_data, dict) and all(k in metric_data for k in ['mean', 'std', 'min', 'max']):
                        writer.writerow([
                            metric,
                            round(metric_data['mean'], 2),
                            round(metric_data['std'], 2),
                            round(metric_data['min'], 2),
                            round(metric_data['max'], 2)
                        ])
            writer.writerow([])
            
            # Write Order Status Summary section
            writer.writerow(['Order Status Summary'])
            writer.writerow(['Status', 'Count'])
            if 'order_status' in data:
                for status, count in data['order_status'].items():
                    writer.writerow([status, count])
            writer.writerow([])
            
            # Write Order Lifecycle Data section
            writer.writerow(['Order Lifecycle Data'])
            
            if 'order_lifecycle' in data and data['order_lifecycle']:
                # IMPORTANT: Order ID format must always be ORD_<random 8-digit number>
                # Example: ORD_12345678 (NO LETTERS allowed in the number portion)
                fields = [
                    'Event Index', 'Order ID', 'Event Date', 'Current Status',
                    'Current Location', 'Production Time', 'Transit Time', 'Delay Time',
                    'Expected Delivery', 'Actual Delivery', 'Transportation Mode',
                    'Source Region', 'Destination Region', 'Is Delayed', 'Is On Time'
                ]
                
                # Write header
                writer.writerow(fields)
                
                # Flatten and sort all events by simulation day and created_at
                all_events = []
                for order_id, events in data['order_lifecycle'].items():
                    # Generate a new 8-digit random number for the order ID
                    new_order_id = f"ORD_{random.randint(10000000, 99999999)}"
                    for event in events:
                        event_copy = event.copy()
                        event_copy['Order ID'] = new_order_id
                        all_events.append(event_copy)
                
                # Sort events by simulation day and created_at
                all_events.sort(key=lambda x: (x['simulation_day'], x['created_at']))
                
                # Write data rows with global event counter
                for event_index, event in enumerate(all_events):
                    row = [
                        event_index,
                        event['Order ID'],
                        event.get('created_at', 'NA'),
                        event.get('current_status', 'NA'),
                        event.get('current_location', 'NA'),
                        event.get('production_time', 0),
                        event.get('transit_time', 0),
                        event.get('delay_time', 0),
                        event.get('expected_delivery', 'NA'),
                        event.get('actual_delivery', 'NA'),
                        event.get('transportation_mode', 'NA'),
                        event.get('source_region', 'NA'),
                        event.get('destination_region', 'NA'),
                        event.get('is_delayed', False),
                        event.get('is_on_time', None)
                    ]
                    writer.writerow(row)

    def _save_results_to_csv(self, results: Dict[str, Any], results_df: pd.DataFrame, scenario_name: str):
        """Save simulation results to CSV files."""
        # Create timestamped simulation directory
        results_dir = os.path.join("test_results", f"simulation_{self.simulation_id}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save results with full order tracking
        results_file = os.path.join(results_dir, f"{scenario_name}_results.csv")
        self._dict_to_csv(results, results_file)
        
        # Save DataFrames to CSV if they exist
        if not results_df.empty:
            results_df.to_csv(os.path.join(results_dir, f"{scenario_name}_metrics.csv"))

    def _analyze_interactions(self, baseline_df: pd.DataFrame, improved_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze agent interactions and return key metrics."""
        analysis = {
            'baseline': {},
            'improved': {},
            'comparison': {}
        }
        
        if not baseline_df.empty:
            analysis['baseline'] = {
                'total_interactions': len(baseline_df),
                'unique_agents': baseline_df['agent_name'].nunique(),
                'interaction_types': baseline_df['action_type'].value_counts().to_dict(),
                'avg_interactions_per_agent': len(baseline_df) / baseline_df['agent_name'].nunique()
            }
            
        if not improved_df.empty:
            analysis['improved'] = {
                'total_interactions': len(improved_df),
                'unique_agents': improved_df['agent_name'].nunique(),
                'interaction_types': improved_df['action_type'].value_counts().to_dict(),
                'avg_interactions_per_agent': len(improved_df) / improved_df['agent_name'].nunique()
            }
            
        if not baseline_df.empty and not improved_df.empty:
            analysis['comparison'] = {
                'total_interactions_change': analysis['improved']['total_interactions'] - analysis['baseline']['total_interactions'],
                'unique_agents_change': analysis['improved']['unique_agents'] - analysis['baseline']['unique_agents'],
                'avg_interactions_change': analysis['improved']['avg_interactions_per_agent'] - analysis['baseline']['avg_interactions_per_agent']
            }
            
        return analysis

    def _save_results_to_markdown(self, baseline_results, improved_results, baseline_df, improved_df):
        """Save simulation results to a markdown file."""
        # Use the existing timestamp and directory
        results_dir = os.path.join("test_results", f"simulation_{self.simulation_id}")
        os.makedirs(results_dir, exist_ok=True)

        with open(os.path.join(results_dir, "results_summary.md"), 'w') as f:
            f.write("# Supply Chain Simulation Results\n\n")
            
            # Core Metrics Comparison Table
            f.write("## Core Metrics Comparison\n\n")
            f.write("| Metric | Baseline | Improved | Absolute Change | Relative Change |\n")
            f.write("|--------|-----------|-----------|-----------------|----------------|\n")
            
            # List of core metrics to compare
            core_metrics = [
                'service_level', 'risk_exposure', 'flexibility_score', 
                'quality_score', 'resilience_score', 'lead_time',
                'recovery_time', 'supplier_risk', 'transportation_risk'
            ]
            
            for metric in core_metrics:
                baseline_value = baseline_results.get(metric, {}).get('mean', 0)
                improved_value = improved_results.get(metric, {}).get('mean', 0)
                abs_change = improved_value - baseline_value
                rel_change = (abs_change / baseline_value * 100) if baseline_value != 0 else float('inf')
                
                # Format the values
                baseline_str = f"{baseline_value:.3f}"
                improved_str = f"{improved_value:.3f}"
                abs_change_str = f"{abs_change:+.3f}"
                rel_change_str = f"{rel_change:+.1f}%" if rel_change != float('inf') else "N/A"
                
                metric_name = metric.replace('_', ' ').title()
                f.write(f"| {metric_name} | {baseline_str} | {improved_str} | {abs_change_str} | {rel_change_str} |\n")
            
            # Order Status Comparison
            f.write("\n## Order Status Comparison\n\n")
            f.write("| Status | Baseline | Improved | Change |\n")
            f.write("|--------|-----------|-----------|--------|\n")
            
            # Get all possible statuses from both scenarios
            all_statuses = set()
            if 'order_status' in baseline_results:
                all_statuses.update(baseline_results['order_status'].keys())
            if 'order_status' in improved_results:
                all_statuses.update(improved_results['order_status'].keys())
            
            for status in sorted(all_statuses):
                baseline_count = baseline_results.get('order_status', {}).get(status, 0)
                improved_count = improved_results.get('order_status', {}).get(status, 0)
                change = improved_count - baseline_count
                
                status_name = status.replace('_', ' ').title()
                f.write(f"| {status_name} | {baseline_count} | {improved_count} | {change:+d} |\n")
            
            # Summary Statistics
            f.write("\n## Summary Statistics\n\n")
            
            # Calculate total orders for both scenarios
            baseline_total = sum(baseline_results.get('order_status', {}).values())
            improved_total = sum(improved_results.get('order_status', {}).values())
            
            f.write(f"- Total Orders Processed:\n")
            f.write(f"  - Baseline: {baseline_total}\n")
            f.write(f"  - Improved: {improved_total}\n")
            f.write(f"  - Difference: {improved_total - baseline_total:+d}\n\n")
            
            # Add completion rate
            baseline_completed = baseline_results.get('order_status', {}).get('delivered', 0)
            improved_completed = improved_results.get('order_status', {}).get('delivered', 0)
            
            baseline_completion_rate = (baseline_completed / baseline_total * 100) if baseline_total > 0 else 0
            improved_completion_rate = (improved_completed / improved_total * 100) if improved_total > 0 else 0
            
            f.write(f"- Order Completion Rate:\n")
            f.write(f"  - Baseline: {baseline_completion_rate:.1f}%\n")
            f.write(f"  - Improved: {improved_completion_rate:.1f}%\n")
            f.write(f"  - Difference: {improved_completion_rate - baseline_completion_rate:+.1f}%\n")

        # Save DataFrames to CSV if they exist
        if baseline_df is not None and not baseline_df.empty:
            baseline_df.to_csv(os.path.join(results_dir, "baseline_metrics.csv"))
        if improved_df is not None and not improved_df.empty:
            improved_df.to_csv(os.path.join(results_dir, "improved_metrics.csv"))

    def _print_results(self, results, scenario_name):
        """Print detailed simulation results."""
        print(f"\n{scenario_name} Scenario Results:")
        print("-" * 50)
        
        # Print daily metrics (raw order status)
        print("\nDaily Metrics (Order Status):")
        order_status = results.get('order_status', {})
        print(f"Created Orders: {order_status.get('created', 0)}")
        print(f"In Production: {order_status.get('in_production', 0)}")
        print(f"Ready for Shipping: {order_status.get('ready_for_shipping', 0)}")
        print(f"In Transit: {order_status.get('in_transit', 0)}")
        print(f"Delayed: {order_status.get('delayed', 0)}")
        print(f"Delivered: {order_status.get('delivered', 0)}")
        
        # Calculate and print total orders
        total = sum(order_status.get(status, 0) for status in 
                   ['created', 'in_production', 'ready_for_shipping', 'in_transit', 'delayed', 'delivered'])
        print(f"Total Orders: {total}")
        
        # Print normalized metrics
        print("\nNormalized Core Metrics:")
        for metric in ['service_level', 'risk_exposure', 'flexibility_score', 'quality_score', 'resilience_score']:
            if metric in results:
                values = results[metric]
                print(f"{metric.replace('_', ' ').title()}:")
                print(f"  Mean: {values['mean']:.3f}")
                print(f"  Std Dev: {values['std']:.3f}")
                print(f"  Min: {values['min']:.3f}")
                print(f"  Max: {values['max']:.3f}")

    def _save_interactions_to_markdown(self, baseline_interactions, improved_interactions, results_dir):
        """Save agent interactions to a markdown file with a structured format."""
        with open(os.path.join(results_dir, "agent_interactions.md"), 'w') as f:
            f.write("# Supply Chain Agent Interactions\n\n")
            
            # Function to process interactions for a scenario
            def process_scenario_interactions(interactions, scenario_name):
                f.write(f"## {scenario_name} Scenario\n\n")
                
                # Group interactions by agent
                agent_interactions = {}
                for interaction in interactions:
                    agent_name = interaction.get('agent_name', 'Unknown Agent')
                    if agent_name not in agent_interactions:
                        agent_interactions[agent_name] = []
                    agent_interactions[agent_name].append(interaction)
                
                # Write interactions for each agent
                for agent_name, agent_data in sorted(agent_interactions.items()):
                    f.write(f"### {agent_name}\n\n")
                    f.write("| Timestamp | Action Type | Target | Content |\n")
                    f.write("|-----------|-------------|---------|----------|\n")
                    
                    # Sort interactions by timestamp
                    sorted_interactions = sorted(agent_data, key=lambda x: x.get('timestamp', ''))
                    
                    for interaction in sorted_interactions:
                        timestamp = interaction.get('timestamp', '')
                        action_type = interaction.get('action_type', '')
                        target = interaction.get('target', '')
                        content = interaction.get('content', '')
                        
                        # Clean and format content for markdown table
                        content = content.replace('\n', ' ').replace('|', '\\|')
                        if len(content) > 100:
                            content = content[:97] + '...'
                            
                        f.write(f"| {timestamp} | {action_type} | {target} | {content} |\n")
                    
                    f.write("\n")
                
                # Add interaction statistics
                f.write("### Interaction Statistics\n\n")
                total_interactions = len(interactions)
                unique_agents = len(agent_interactions)
                action_types = {}
                for interaction in interactions:
                    action = interaction.get('action_type', 'Unknown')
                    action_types[action] = action_types.get(action, 0) + 1
                
                f.write(f"- Total Interactions: {total_interactions}\n")
                f.write(f"- Unique Agents: {unique_agents}\n")
                f.write("- Action Type Distribution:\n")
                for action, count in sorted(action_types.items()):
                    percentage = (count / total_interactions) * 100
                    f.write(f"  - {action}: {count} ({percentage:.1f}%)\n")
                f.write("\n")
            
            # Process both scenarios
            if baseline_interactions:
                process_scenario_interactions(baseline_interactions, "Baseline")
            
            if improved_interactions:
                process_scenario_interactions(improved_interactions, "Improved")
            
            # Add comparison section if both scenarios exist
            if baseline_interactions and improved_interactions:
                f.write("## Scenario Comparison\n\n")
                baseline_count = len(baseline_interactions)
                improved_count = len(improved_interactions)
                interaction_change = improved_count - baseline_count
                percentage_change = ((improved_count - baseline_count) / baseline_count * 100) if baseline_count > 0 else float('inf')
                
                f.write("### Interaction Volume\n\n")
                f.write(f"- Baseline Interactions: {baseline_count}\n")
                f.write(f"- Improved Interactions: {improved_count}\n")
                f.write(f"- Change: {interaction_change:+d} ({percentage_change:+.1f}%)\n\n")
                
                # Compare action type distributions
                f.write("### Action Type Changes\n\n")
                f.write("| Action Type | Baseline | Improved | Change |\n")
                f.write("|-------------|-----------|-----------|--------|\n")
                
                # Get all action types
                action_types = set()
                for interaction in baseline_interactions + improved_interactions:
                    action_types.add(interaction.get('action_type', 'Unknown'))
                
                # Count occurrences in each scenario
                baseline_actions = {}
                improved_actions = {}
                for action in action_types:
                    baseline_actions[action] = sum(1 for i in baseline_interactions if i.get('action_type') == action)
                    improved_actions[action] = sum(1 for i in improved_interactions if i.get('action_type') == action)
                
                # Write comparison table
                for action in sorted(action_types):
                    baseline_count = baseline_actions.get(action, 0)
                    improved_count = improved_actions.get(action, 0)
                    change = improved_count - baseline_count
                    f.write(f"| {action} | {baseline_count} | {improved_count} | {change:+d} |\n")

    def _save_simulation_results(self, results: Dict[str, Any], scenario_name: str):
        """Save simulation results using TestArtifactGenerator."""
        # Convert results to DataFrames
        metrics_df = pd.DataFrame(results.get('metrics', {}))
        order_lifecycle_df = pd.DataFrame(results.get('order_lifecycle', {}))
        agent_interactions = results.get('agent_interactions', [])
        
        # Save metrics summary
        metrics_filename = f"{scenario_name}_results.csv"
        self.artifact_generator.save_metrics_summary(
            metrics_df,
            os.path.join(self.artifact_dir, metrics_filename)
        )
        
        # Save order lifecycle data
        lifecycle_filename = f"{scenario_name}_order_lifecycle.csv"
        self.artifact_generator.save_order_lifecycle(
            order_lifecycle_df,
            os.path.join(self.artifact_dir, lifecycle_filename)
        )
        
        # Save agent interactions
        interactions_filename = "agent_interactions.md"
        self.artifact_generator.save_agent_interactions(
            agent_interactions,
            os.path.join(self.artifact_dir, interactions_filename)
        )

    def test_monte_carlo_simulation(self):
        """Test Monte Carlo simulation with multiple iterations."""
        # Run baseline simulation
        baseline_results = run_monte_carlo_simulation(
            self.world,
            self.config,
            scenario_name="baseline"
        )
        
        # Save baseline results
        self._save_simulation_results(baseline_results, "baseline")
        
        # Run improved simulation with optimized parameters
        improved_config = self.config.copy()
        improved_config['supplier']['reliability'] = 0.95  # Increase supplier reliability
        improved_config['logistics']['speed'] = 1.2  # Increase logistics speed
        
        improved_results = run_monte_carlo_simulation(
            self.world,
            improved_config,
            scenario_name="improved"
        )
        
        # Save improved results
        self._save_simulation_results(improved_results, "improved")
        
        # Generate a summary markdown file comparing the scenarios
        summary_path = os.path.join(self.artifact_dir, "results_summary.md")
        with open(summary_path, "w") as f:
            f.write("# Monte Carlo Simulation Results Summary\n\n")
            f.write("## Baseline Scenario\n")
            f.write(self._format_results_summary(baseline_results))
            f.write("\n## Improved Scenario\n")
            f.write(self._format_results_summary(improved_results))
            f.write("\n## Comparison\n")
            f.write(self._format_comparison(baseline_results, improved_results))

    def _format_results_summary(self, results: Dict[str, Any]) -> str:
        """Format results for markdown summary."""
        summary = []
        metrics = results.get('metrics', {})
        
        if metrics:
            summary.append("### Key Metrics")
            summary.append("- Average Order Completion Time: {:.2f}".format(
                metrics.get('avg_completion_time', 0)
            ))
            summary.append("- Order Success Rate: {:.2%}".format(
                metrics.get('success_rate', 0)
            ))
            summary.append("- Average Cost per Order: {:.2f}".format(
                metrics.get('avg_cost', 0)
            ))
        
        return "\n".join(summary)

    def _format_comparison(self, baseline: Dict[str, Any], improved: Dict[str, Any]) -> str:
        """Format comparison between baseline and improved results."""
        baseline_metrics = baseline.get('metrics', {})
        improved_metrics = improved.get('metrics', {})
        
        comparison = ["### Metrics Comparison"]
        
        for metric in ['avg_completion_time', 'success_rate', 'avg_cost']:
            baseline_val = baseline_metrics.get(metric, 0)
            improved_val = improved_metrics.get(metric, 0)
            diff = improved_val - baseline_val
            diff_pct = (diff / baseline_val * 100) if baseline_val else 0
            
            comparison.append(f"- {metric.replace('_', ' ').title()}:")
            comparison.append(f"  - Baseline: {baseline_val:.2f}")
            comparison.append(f"  - Improved: {improved_val:.2f}")
            comparison.append(f"  - Change: {diff_pct:+.1f}%")
        
        return "\n".join(comparison)

if __name__ == '__main__':
    unittest.main(verbosity=2) 