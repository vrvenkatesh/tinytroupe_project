import os
import csv
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

from tinytroupe.extraction import ResultsExtractor, ResultsReducer
from tinytroupe.agent import TinyPerson
from tinytroupe.environment import TinyWorld

class TestArtifactGenerator:
    """Helper class to generate test artifacts."""

    def __init__(self, simulation_id: str):
        """Initialize the test artifact generator.
        
        Args:
            simulation_id (str): Unique identifier for the simulation run.
        """
        self.simulation_id = simulation_id
        self.results_dir = os.path.join("test_results", f"simulation_{simulation_id}")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize extractors
        self.results_extractor = ResultsExtractor()
        self.results_reducer = ResultsReducer()

    def save_metrics_summary(self, metrics: Dict[str, Any], scenario_name: str) -> None:
        """Save metrics summary to CSV file.
        
        Args:
            metrics (Dict[str, Any]): Dictionary containing metrics data.
            scenario_name (str): Name of the scenario (e.g., 'baseline', 'improved').
        """
        output_file = os.path.join(self.results_dir, f"{scenario_name}_metrics.csv")
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write Core Metrics section
            writer.writerow(['Core Metrics'])
            writer.writerow(['Metric', 'Mean', 'Std Dev', 'Min', 'Max'])
            
            if 'metrics' in metrics:
                for metric_name, metric_data in metrics['metrics'].items():
                    writer.writerow([
                        metric_name,
                        round(metric_data['mean'], 3),
                        round(metric_data['std'], 3),
                        round(metric_data['min'], 3),
                        round(metric_data['max'], 3)
                    ])
            
            writer.writerow([])  # Empty row for separation
            
            # Write Order Status Summary section
            writer.writerow(['Order Status Summary'])
            writer.writerow(['Status', 'Count'])
            if 'order_status' in metrics:
                for status, count in metrics['order_status'].items():
                    writer.writerow([status, count])

    def save_order_lifecycle(self, world: TinyWorld, scenario_name: str) -> None:
        """Save complete order lifecycle data to CSV file.
        
        Args:
            world (TinyWorld): The simulation world containing order data.
            scenario_name (str): Name of the scenario (e.g., 'baseline', 'improved').
        """
        output_file = os.path.join(self.results_dir, f"{scenario_name}_order_lifecycle.csv")
        
        # Get all orders from the world state
        active_orders = world.state.get('active_orders', [])
        completed_orders = world.state.get('completed_orders', [])
        all_orders = active_orders + completed_orders
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'Event Index', 'Order ID', 'Event Date', 'Current Status',
                'Current Location', 'Production Time', 'Transit Time', 'Delay Time',
                'Expected Delivery', 'Actual Delivery', 'Transportation Mode',
                'Source Region', 'Destination Region', 'Simulation Day', 'Is Delayed',
                'Is On Time'
            ])
            
            # Write order lifecycle events
            event_index = 0
            for order in all_orders:
                # Extract order data with proper error handling
                order_data = {
                    'id': getattr(order, 'id', f'ORD_{event_index:08d}'),
                    'creation_time': getattr(order, 'creation_time', datetime.now()),
                    'status': getattr(order, 'status', 'NEW'),
                    'current_location': getattr(order, 'current_location', 'UNKNOWN'),
                    'production_time': float(getattr(order, 'production_time', 0)),
                    'transit_time': float(getattr(order, 'transit_time', 0)),
                    'delay_time': float(getattr(order, 'delay_time', 0)),
                    'expected_delivery_time': getattr(order, 'expected_delivery_time', None),
                    'actual_delivery_time': getattr(order, 'actual_delivery_time', None),
                    'transportation_mode': getattr(order, 'transportation_mode', 'UNKNOWN'),
                    'source_region': getattr(order, 'source_region', 'UNKNOWN'),
                    'destination_region': getattr(order, 'destination_region', 'UNKNOWN'),
                    'simulation_day': int(getattr(order, 'simulation_day', event_index % 10))
                }
                
                # Format timestamps
                creation_time_str = order_data['creation_time'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(order_data['creation_time'], datetime) else str(order_data['creation_time'])
                expected_delivery_str = order_data['expected_delivery_time'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(order_data['expected_delivery_time'], datetime) else 'NA'
                actual_delivery_str = order_data['actual_delivery_time'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(order_data['actual_delivery_time'], datetime) else 'NA'
                
                # Calculate derived fields - Updated logic for mutual exclusivity
                is_delayed = False
                is_on_time = False
                
                if actual_delivery_str != 'NA':
                    # Order has been delivered - check if it was delayed
                    is_delayed = bool(order_data['delay_time'] > 0)
                    is_on_time = not is_delayed
                elif expected_delivery_str != 'NA':
                    # Order not delivered yet - check if it's past expected delivery
                    expected_time = order_data['expected_delivery_time']
                    current_time = order_data['creation_time']
                    if isinstance(expected_time, datetime) and isinstance(current_time, datetime):
                        is_delayed = current_time > expected_time
                        is_on_time = not is_delayed
                
                # Write row with proper type handling
                row = [
                    event_index,
                    str(order_data['id']),
                    creation_time_str,
                    str(order_data['status'].value if hasattr(order_data['status'], 'value') else order_data['status']),
                    str(order_data['current_location'].value if hasattr(order_data['current_location'], 'value') else order_data['current_location']),
                    order_data['production_time'],
                    order_data['transit_time'],
                    order_data['delay_time'],
                    expected_delivery_str,
                    actual_delivery_str,
                    str(order_data['transportation_mode'].value if hasattr(order_data['transportation_mode'], 'value') else order_data['transportation_mode']),
                    str(order_data['source_region'].value if hasattr(order_data['source_region'], 'value') else order_data['source_region']),
                    str(order_data['destination_region'].value if hasattr(order_data['destination_region'], 'value') else order_data['destination_region']),
                    order_data['simulation_day'],
                    is_delayed,
                    is_on_time
                ]
                writer.writerow(row)
                event_index += 1

    def save_agent_interactions(self, world: TinyWorld, scenario_name: str) -> None:
        """Save agent interaction data to CSV file.
        
        Args:
            world (TinyWorld): The simulation world containing agent data.
            scenario_name (str): Name of the scenario (e.g., 'baseline', 'improved').
        """
        output_file = os.path.join(self.results_dir, f"{scenario_name}_agent_interactions.csv")
        
        # Get all agents from the world state
        agents = world.state.get('agents', [])
        interactions = []
        
        # Collect all interactions
        for agent in agents:
            agent_interactions = getattr(agent, 'interactions', [])
            for interaction in agent_interactions:
                if interaction:  # Skip None or empty interactions
                    interactions.append({
                        'agent_id': str(getattr(agent, 'id', 'unknown')),
                        'agent_type': getattr(agent, 'agent_type', type(agent).__name__),  # Get explicit type or class name
                        'interaction_type': str(getattr(interaction, 'type', 'unknown')),
                        'timestamp': getattr(interaction, 'timestamp', datetime.now()),
                        'target_agent': str(getattr(interaction, 'target_agent', 'unknown')),
                        'order_id': str(getattr(interaction, 'order_id', 'unknown')),
                        'status': getattr(interaction, 'status', 'unknown'),
                        'success': bool(getattr(interaction, 'success', False)),
                        'message': str(getattr(interaction, 'message', '')),
                        'simulation_day': int(getattr(interaction, 'simulation_day', 0))
                    })
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'Interaction ID', 'Agent ID', 'Agent Type', 'Interaction Type',
                'Timestamp', 'Target Agent', 'Order ID', 'Status', 'Success',
                'Message', 'Simulation Day'
            ])
            
            # Write interaction data with proper formatting
            for idx, interaction in enumerate(interactions):
                timestamp_str = interaction['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(interaction['timestamp'], datetime) else str(interaction['timestamp'])
                
                row = [
                    f'INT_{idx:08d}',
                    interaction['agent_id'],
                    interaction['agent_type'],
                    interaction['interaction_type'],
                    timestamp_str,
                    interaction['target_agent'],
                    interaction['order_id'],
                    str(interaction['status'].value if hasattr(interaction['status'], 'value') else interaction['status']),
                    str(interaction['success']),
                    interaction['message'],
                    interaction['simulation_day']
                ]
                writer.writerow(row)

    def generate_artifacts(self, world: TinyWorld, metrics: Dict[str, Any], scenario_name: str = "baseline") -> None:
        """Generate all test artifacts for a simulation run.
        
        Args:
            world (TinyWorld): The simulation world.
            metrics (Dict[str, Any]): Dictionary containing metrics data.
            scenario_name (str, optional): Name of the scenario. Defaults to "baseline".
        """
        # Save metrics summary
        self.save_metrics_summary(metrics, scenario_name)
        
        # Save order lifecycle data
        self.save_order_lifecycle(world, scenario_name)
        
        # Save agent interactions
        self.save_agent_interactions(world, scenario_name) 