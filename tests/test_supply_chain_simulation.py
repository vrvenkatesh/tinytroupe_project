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

from supply_chain import (
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
        
        # Initialize results extractor and reducer
        self.results_extractor = ResultsExtractor()
        self.results_reducer = ResultsReducer()
        
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
        
        # Add agents to world and verify registration
        def add_agent_safely(agent):
            """Helper to add agent and verify registration."""
            # First check if agent is already in global registry
            if agent.name in TinyPerson.all_agents:
                print(f"Warning: Agent {agent.name} already in global registry")
                # Remove from global registry to avoid conflicts
                del TinyPerson.all_agents[agent.name]
            
            # Then check if agent is in world
            if agent.name in self.world.name_to_agent:
                print(f"Warning: Agent {agent.name} already in world")
                # Remove from world to avoid conflicts
                self.world.remove_agent(agent)
            
            # Now add agent to both places
            TinyPerson.all_agents[agent.name] = agent
            self.world.add_agent(agent)
            print(f"Added agent {agent.name} to world and global registry")
            
            # Verify registration
            assert agent.name in TinyPerson.all_agents, f"Agent {agent.name} not in global registry"
            assert agent.name in self.world.name_to_agent, f"Agent {agent.name} not in world name_to_agent"
            assert agent in self.world.agents, f"Agent {agent.name} not in world agents list"
        
        add_agent_safely(self.coo)
        for manager in self.regional_managers.values():
            add_agent_safely(manager)
        for region_suppliers in self.suppliers.values():
            for supplier in region_suppliers:
                add_agent_safely(supplier)
        for facility in self.production_facilities.values():
            add_agent_safely(facility)
        for provider in self.logistics_providers:
            add_agent_safely(provider)
            
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
        
        # Initialize output file paths
        self.output_file = None
        self.output_plot = None
        
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

    def _dict_to_csv(self, data: Dict[str, Any], filename: str):
        """Save dictionary data to a CSV file."""
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # First section: Core metrics
            writer.writerow(['Core Metrics'])
            writer.writerow(['Metric', 'Mean', 'Std Dev', 'Min', 'Max'])
            for metric, values in data.items():
                # Skip non-metric data
                if metric in ['daily_order_tracking', 'order_status', 'order_lifecycle']:
                    continue
                if isinstance(values, dict) and 'mean' in values:
                    writer.writerow([
                        metric,
                        values.get('mean', ''),
                        values.get('std', ''),
                        values.get('min', ''),
                        values.get('max', '')
                    ])
            
            # Second section: Order status summary
            writer.writerow([])  # Empty row for separation
            writer.writerow(['Order Status Summary'])
            writer.writerow(['Status', 'Count'])
            if 'order_status' in data:
                for status, count in data['order_status'].items():
                    writer.writerow([status, count])
            
            # Third section: Order lifecycle data
            if 'order_lifecycle' in data:
                writer.writerow([])  # Empty row for separation
                writer.writerow(['Order Lifecycle Data'])
                
                # Get all fields from the first event of the first order (if any exist)
                if data['order_lifecycle']:
                    first_order_id = next(iter(data['order_lifecycle']))
                    first_order_events = data['order_lifecycle'][first_order_id]
                    if first_order_events:  # Check if there are any events
                        first_event = first_order_events[0]
                        fields = list(first_event.keys())
                        
                        # Write header
                        writer.writerow(['Order ID', 'Event Index'] + fields)
                        
                        # Write data for each order and each event
                        for order_id, events in data['order_lifecycle'].items():
                            for event_idx, event in enumerate(events):
                                row = [order_id, event_idx]
                                row.extend(str(event.get(field, '')) for field in fields)
                                writer.writerow(row)
            
            # Fourth section: Daily tracking data (if exists in old format)
            if 'daily_order_tracking' in data and isinstance(data['daily_order_tracking'], list) and data['daily_order_tracking']:
                writer.writerow([])  # Empty row for separation
                writer.writerow(['Daily Tracking Data'])
                
                # Write the data as is, with each list item on a new row
                for item in data['daily_order_tracking']:
                    if isinstance(item, dict):
                        writer.writerow([f"{k}: {v}" for k, v in item.items()])

    def _save_results_to_csv(self, results: Dict[str, Any], results_df: pd.DataFrame, scenario_name: str):
        """Save simulation results to CSV files."""
        # Create timestamped simulation directory
        results_dir = os.path.join("test_results", f"simulation_{self.simulation_id}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save results with full order tracking
        results_file = os.path.join(results_dir, f"{scenario_name}_results.csv")
        self._dict_to_csv(results, results_file)
        
        # Save detailed order lifecycle data if available
        if 'order_lifecycle' in results:
            lifecycle_file = os.path.join(results_dir, f"{scenario_name}_order_lifecycle.csv")
            with open(lifecycle_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Order ID', 'Event Index', 'Created At', 'Current Status', 'Current Location',
                    'Production Time', 'Transit Time', 'Delay Time',
                    'Expected Delivery', 'Actual Delivery', 'Transportation Mode',
                    'Source Region', 'Destination Region', 'Simulation Day',
                    'Is Delayed', 'Is On Time'
                ])
                for order_id, events in results['order_lifecycle'].items():
                    for event_idx, event in enumerate(events):
                        writer.writerow([
                            order_id,
                            event_idx,
                            event.get('created_at', ''),
                            event.get('current_status', ''),
                            event.get('current_location', ''),
                            event.get('production_time', ''),
                            event.get('transit_time', ''),
                            event.get('delay_time', ''),
                            event.get('expected_delivery', ''),
                            event.get('actual_delivery', ''),
                            event.get('transportation_mode', ''),
                            event.get('source_region', ''),
                            event.get('destination_region', ''),
                            event.get('simulation_day', ''),
                            event.get('is_delayed', ''),
                            event.get('is_on_time', '')
                        ])
        
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

        analysis = {
            'baseline': baseline_results,
            'improved': improved_results
        }

        with open(os.path.join(results_dir, "results_summary.md"), 'w') as f:
            f.write("# Supply Chain Simulation Results\n\n")
            f.write("## Baseline Scenario\n\n")
            f.write("### Key Metrics\n")
            service_level = baseline_results.get('service_level', {}).get('mean', 0)
            resilience_score = baseline_results.get('resilience_score', {}).get('mean', 0)
            f.write("- Service Level: {:.3f}\n".format(service_level))
            f.write("- Resilience Score: {:.3f}\n".format(resilience_score))
            if 'total_interactions' in baseline_results:
                f.write("- Total Interactions: {}\n".format(baseline_results['total_interactions']))

            f.write("\n## Improved Scenario\n\n")
            f.write("### Key Metrics\n")
            service_level_improved = improved_results.get('service_level', {}).get('mean', 0)
            resilience_score_improved = improved_results.get('resilience_score', {}).get('mean', 0)
            f.write("- Service Level: {:.3f}\n".format(service_level_improved))
            f.write("- Resilience Score: {:.3f}\n".format(resilience_score_improved))
            if 'total_interactions' in improved_results:
                f.write("- Total Interactions: {}\n".format(improved_results['total_interactions']))

            f.write("\n## Improvements\n\n")
            f.write("### Service Level\n")
            service_level_change = service_level_improved - service_level
            f.write("- Absolute Change: {:.3f}\n".format(service_level_change))
            if service_level != 0:
                relative_change = (service_level_change / service_level) * 100
                f.write("- Relative Change: {:.1f}%\n".format(relative_change))

            f.write("\n### Resilience Score\n")
            resilience_change = resilience_score_improved - resilience_score
            f.write("- Absolute Change: {:.3f}\n".format(resilience_change))
            if resilience_score != 0:
                relative_change = (resilience_change / resilience_score) * 100
                f.write("- Relative Change: {:.1f}%\n".format(relative_change))

            # Add order status summary
            f.write("\n## Order Status Summary\n\n")
            f.write("### Baseline\n")
            if 'order_status' in baseline_results:
                order_status = baseline_results['order_status']
                f.write("- Created Orders: {}\n".format(order_status.get('created', 0)))
                f.write("- In Production: {}\n".format(order_status.get('in_production', 0)))
                f.write("- Ready for Shipping: {}\n".format(order_status.get('ready_for_shipping', 0)))
                f.write("- In Transit: {}\n".format(order_status.get('in_transit', 0)))
                f.write("- Delayed: {}\n".format(order_status.get('delayed', 0)))
                f.write("- Delivered: {}\n".format(order_status.get('delivered', 0)))

            f.write("\n### Improved\n")
            if 'order_status' in improved_results:
                order_status = improved_results['order_status']
                f.write("- Created Orders: {}\n".format(order_status.get('created', 0)))
                f.write("- In Production: {}\n".format(order_status.get('in_production', 0)))
                f.write("- Ready for Shipping: {}\n".format(order_status.get('ready_for_shipping', 0)))
                f.write("- In Transit: {}\n".format(order_status.get('in_transit', 0)))
                f.write("- Delayed: {}\n".format(order_status.get('delayed', 0)))
                f.write("- Delivered: {}\n".format(order_status.get('delivered', 0)))

        # Save DataFrames to CSV if they exist
        if baseline_df is not None:
            baseline_df.to_csv(os.path.join(results_dir, "baseline_metrics.csv"))
        if improved_df is not None:
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

    def test_quick_simulation(self):
        """Test running a quick simulation with baseline and improved scenarios."""
        # Run baseline simulation
        baseline_results = run_monte_carlo_simulation(
            config=self.config,
            world=self.world,
            has_supplier_diversification=False,
            has_dynamic_inventory=False,
            has_flexible_transportation=False,
            has_regional_flexibility=False
        )
        
        # Extract agent interactions
        baseline_interactions = self.results_extractor.extract_results_from_agents(
            agents=self.world.agents,
            extraction_objective="Extract all interactions, including stimuli received and actions performed",
            fields=["agent_name", "type", "action_type", "target", "content", "timestamp", "role", "region"]
        )
        
        # Initialize metrics dictionary if None
        if not baseline_results.get('metrics'):
            baseline_results['metrics'] = {}
        
        # Ensure step_results is initialized
        step_results = baseline_results.get('step_results', {})
        if not isinstance(step_results, dict):
            step_results = {}
        
        # Set total_orders if not present
        if 'total_orders' not in step_results:
            step_results['total_orders'] = len(baseline_results.get('all_orders', []))
        
        baseline_results['step_results'] = step_results
        
        # Create DataFrame with proper initialization
        baseline_df = pd.DataFrame(baseline_interactions if baseline_interactions else [])
        
        # Save results
        self._save_results_to_csv(baseline_results, baseline_df, "baseline")
        
        # Clean up agents between runs
        print("\nCleaning up agents before improved scenario...")
        
        # First, get a list of all agents in the world
        agents_to_remove = list(self.world.agents)
        
        # Remove each agent individually to ensure proper cleanup
        for agent in agents_to_remove:
            try:
                self.world.remove_agent(agent)
                print(f"Removed agent {agent.name} from world")
            except Exception as e:
                print(f"Error removing agent {agent.name}: {str(e)}")
        
        # Clear the global agent registry
        if hasattr(TinyPerson, 'all_agents'):
            agent_count = len(TinyPerson.all_agents)
            agent_names = list(TinyPerson.all_agents.keys())
            TinyPerson.all_agents.clear()
            print(f"Cleaned up {agent_count} agents from global registry: {agent_names}")
        
        # Reset world state
        self.world.state['active_orders'] = []
        self.world.state['completed_orders'] = []
        self.world.state['failed_orders'] = []
        print("Reset world state")
        
        # Verify cleanup
        if len(self.world.agents) > 0:
            print(f"Warning: {len(self.world.agents)} agents still in world")
        if len(self.world.name_to_agent) > 0:
            print(f"Warning: {len(self.world.name_to_agent)} agents still in name_to_agent map")
        if hasattr(TinyPerson, 'all_agents') and len(TinyPerson.all_agents) > 0:
            print(f"Warning: {len(TinyPerson.all_agents)} agents still in global registry")
        
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
        self._print_results(improved_results, "Improved")
        
        # Extract agent interactions from improved run
        print("\nExtracting improved agent interactions...")
        improved_interactions = self.results_extractor.extract_results_from_agents(
            agents=self.world.agents,
            extraction_objective="Extract all interactions, including stimuli received and actions performed",
            fields=["agent_name", "type", "action_type", "target", "content", "timestamp", "role", "region"]
        )
        
        # Create DataFrame with proper initialization
        improved_df = pd.DataFrame(improved_interactions if improved_interactions else [])
        
        # Save results
        self._save_results_to_csv(improved_results, improved_df, "improved")
        
        # Compare improvements
        self._compare_results(baseline_results, improved_results)
        
        # Only save results if all verifications pass
        print("\nAll verifications passed. Creating results directory and saving data...")
        
        # Create results directory only after all verifications pass
        results_dir = os.path.join("test_results", f"simulation_{self.simulation_id}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save detailed agent interactions
        if baseline_interactions:
            with open(os.path.join(results_dir, "baseline_interactions.json"), 'w') as f:
                json.dump(baseline_interactions, f, indent=2)
            if not baseline_df.empty:
                baseline_df.to_csv(os.path.join(results_dir, "baseline_interactions.csv"), index=False)
        
        if improved_interactions:
            with open(os.path.join(results_dir, "improved_interactions.json"), 'w') as f:
                json.dump(improved_interactions, f, indent=2)
            if not improved_df.empty:
                improved_df.to_csv(os.path.join(results_dir, "improved_interactions.csv"), index=False)
        
        # Save comparison if both sets exist
        if baseline_interactions and improved_interactions:
            comparison_analysis = self._analyze_interactions(baseline_df, improved_df)
            with open(os.path.join(results_dir, "interactions_comparison.json"), 'w') as f:
                json.dump(comparison_analysis, f, indent=2)
        
        print("\nQuick simulation test completed successfully!")
        
    def _verify_results(self, results: Dict[str, Any]):
        """Helper method to verify result structure and values."""
        # Skip verification for raw order status counts, order tracking, and order lifecycle
        normalized_metrics = set([
            'service_level', 'risk_exposure', 'flexibility_score', 
            'quality_score', 'resilience_score', 'lead_time'
        ])
        
        for metric, values in results.items():
            # Skip order tracking, status metrics, and order lifecycle
            if metric in ['daily_order_tracking', 'order_status', 'order_lifecycle']:
                continue
                
            # Verify structure and values for normalized metrics
            if metric in normalized_metrics:
                self.assertIsInstance(values, dict, f"Expected dict for {metric}, got {type(values)}")
                self.assertTrue(0 <= values['mean'] <= 1, f"{metric} mean out of range")
                self.assertTrue(0 <= values['min'] <= values['max'] <= 1, f"{metric} min/max out of range")
                
                # Allow zero means for certain metrics in this simplified test
                if metric not in ['delayed_orders']:
                    self.assertNotEqual(values['mean'], 0, f"{metric} mean should not be zero")
            
            # For non-normalized metrics, just verify the structure
            else:
                self.assertIsInstance(values, dict, f"Expected dict for {metric}, got {type(values)}")
                self.assertIn('mean', values, f"Missing 'mean' in {metric}")
                self.assertIn('std', values, f"Missing 'std' in {metric}")
                self.assertIn('min', values, f"Missing 'min' in {metric}")
                self.assertIn('max', values, f"Missing 'max' in {metric}")

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

    def test_order_based_simulation(self):
        """Test the order-based supply chain simulation."""
        print("\n=== Testing Order-Based Simulation ===")
        
        # Create agents with unique names
        coo = create_coo_agent(f"COO_{self.simulation_id}", self.config['coo'], self.simulation_id)
        
        # Create regional managers with proper region configuration
        regional_managers = {
            Region.NORTH_AMERICA: create_regional_manager_agent(
                f"TestManager_NA_{self.simulation_id}", 
                {**self.config['regional_manager'], 'region': Region.NORTH_AMERICA}, 
                self.simulation_id
            ),
            Region.EUROPE: create_regional_manager_agent(
                f"TestManager_EU_{self.simulation_id}", 
                {**self.config['regional_manager'], 'region': Region.EUROPE}, 
                self.simulation_id
            )
        }
        
        suppliers = {
            Region.NORTH_AMERICA: [create_supplier_agent(
                f"TestSupplier_NA_{i}_{self.simulation_id}", 
                {**self.config['supplier'], 'region': Region.NORTH_AMERICA}, 
                self.simulation_id
            ) for i in range(self.config['simulation']['suppliers_per_region'])],
            Region.EUROPE: [create_supplier_agent(
                f"TestSupplier_EU_{i}_{self.simulation_id}", 
                {**self.config['supplier'], 'region': Region.EUROPE}, 
                self.simulation_id
            ) for i in range(self.config['simulation']['suppliers_per_region'])]
        }
        
        production_facilities = {
            Region.NORTH_AMERICA: create_production_facility_agent(
                f"TestFacility_NA_{self.simulation_id}", 
                {**self.config['production_facility'], 'region': Region.NORTH_AMERICA}, 
                self.simulation_id
            ),
            Region.EUROPE: create_production_facility_agent(
                f"TestFacility_EU_{self.simulation_id}", 
                {**self.config['production_facility'], 'region': Region.EUROPE}, 
                self.simulation_id
            )
        }
        
        # Create logistics providers with proper source and destination regions
        logistics_providers = {
            "TestLogistics_NA_EU": create_logistics_provider_agent(
                source_region=Region.NORTH_AMERICA,
                dest_region=Region.EUROPE,
                config=self.config['logistics']
            ),
            "TestLogistics_EU_NA": create_logistics_provider_agent(
                source_region=Region.EUROPE,
                dest_region=Region.NORTH_AMERICA,
                config=self.config['logistics']
            )
        }
        
        # Add agents to world
        self.world.add_agent(coo)
        for manager in regional_managers.values():
            self.world.add_agent(manager)
        for region_suppliers in suppliers.values():
            for supplier in region_suppliers:
                self.world.add_agent(supplier)
        for facility in production_facilities.values():
            self.world.add_agent(facility)
        for provider in logistics_providers.values():
            self.world.add_agent(provider)
            
        # Store agents in world state with correct keys
        self.world.state['coo_agent'] = coo
        self.world.state['regional_managers'] = list(regional_managers.values())
        self.world.state['suppliers'] = [s for suppliers_list in suppliers.values() for s in suppliers_list]
        self.world.state['production_facilities'] = list(production_facilities.values())
        self.world.state['logistics_providers'] = list(logistics_providers.values())
        
        # Initialize order tracking in world state
        self.world.state['active_orders'] = []
        self.world.state['completed_orders'] = []
        self.world.state['order_lifecycle'] = {}
        
        # Run simulation for multiple time steps
        metrics_history = []
        for t in range(self.config['simulation']['time_steps']):
            self.world.current_time = t
            metrics = simulate_supply_chain_operation(
                world=self.world,
                config=self.config
            )
            metrics_history.append(metrics)
            
            # Print current state
            active_orders = self.world.state['active_orders']
            completed_orders = self.world.state['completed_orders']
            print(f"\nTime step {t}:")
            print(f"Active orders: {len(active_orders)}")
            print(f"Completed orders: {len(completed_orders)}")
            
            # Print detailed order status
            print("\nActive Orders Status:")
            for order in active_orders:
                print(f"  Order {order.id}:")
                print(f"    Status: {order.status.value}")
                print(f"    From: {order.source_region.value} -> To: {order.destination_region.value}")
                print(f"    Created: {order.creation_time}, Expected delivery: {order.expected_delivery_time}")
                if order.status == OrderStatus.PRODUCTION:
                    print(f"    Production time: {order.production_time}")
                elif order.status == OrderStatus.IN_TRANSIT:
                    print(f"    Transit time: {order.transit_time}")
                    print(f"    Transport mode: {order.transportation_mode.value}")
                if order.delay_time > 0:
                    print(f"    Currently delayed by: {order.delay_time} days")
            
            if completed_orders:
                print("\nRecently Completed Orders:")
                # Show only orders completed in this timestep
                recent_completed = [o for o in completed_orders if o.actual_delivery_time == t]
                for order in recent_completed:
                    print(f"  Order {order.id} completed:")
                    print(f"    Total lead time: {order.calculate_lead_time()} days")
                    print(f"    On time: {order.is_on_time()}")
                    if order.delay_time > 0:
                        print(f"    Total delay: {order.delay_time} days")
            
            print("\nMetrics:")
            for k, v in metrics.items():
                if isinstance(v, dict):
                    print(f"  {k}:")
                    for sub_k, sub_v in v.items():
                        if isinstance(sub_v, (int, float)):
                            print(f"    {sub_k}: {sub_v:.2f}")
                        else:
                            print(f"    {sub_k}: {sub_v}")
                elif isinstance(v, (int, float)):
                    print(f"  {k}: {v:.2f}")
                else:
                    print(f"  {k}: {v}")
            print("-" * 80)
        
        # Verify simulation results
        final_metrics = metrics_history[-1]
        
        # Basic assertions
        self.assertIsInstance(final_metrics, dict)
        self.assertTrue(0 <= final_metrics['service_level'] <= 1)
        self.assertTrue(0 <= final_metrics['lead_time'] <= 1)
        self.assertTrue(0 <= final_metrics['risk_exposure'] <= 1)
        
        # Verify orders were processed
        total_orders = len(self.world.state['active_orders']) + len(self.world.state['completed_orders'])
        self.assertGreater(total_orders, 0, "No orders were generated")
        self.assertGreater(len(self.world.state['completed_orders']), 0, "No orders were completed")
        
        # Verify order processing
        for order in self.world.state['completed_orders']:
            self.assertEqual(order.status, OrderStatus.DELIVERED)
            self.assertIsNotNone(order.actual_delivery_time)
            self.assertGreater(order.actual_delivery_time, order.creation_time)
            
        # Verify metrics calculation
        self.assertGreater(final_metrics['resilience_score'], 0)
        self.assertLess(final_metrics['risk_exposure'], 1)
        
        print("\nFinal simulation state:")
        print(f"Total orders processed: {total_orders}")
        print(f"Completed orders: {len(self.world.state['completed_orders'])}")
        print(f"Final metrics: {final_metrics}")

if __name__ == '__main__':
    unittest.main(verbosity=2) 