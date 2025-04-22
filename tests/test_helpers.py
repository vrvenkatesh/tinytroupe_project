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
        self.results_dir = os.path.join("test_results", simulation_id)
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
                    try:
                        # Convert all numeric values to float with proper error handling
                        mean_val = float(metric_data.get('mean', 0.0) or 0.0)
                        std_val = float(metric_data.get('std', 0.0) or 0.0)
                        min_val = float(metric_data.get('min', 0.0) or 0.0)
                        max_val = float(metric_data.get('max', 0.0) or 0.0)
                        
                        # Round all values to 3 decimal places
                        writer.writerow([
                            metric_name,
                            round(mean_val, 3),
                            round(std_val, 3),
                            round(min_val, 3),
                            round(max_val, 3)
                        ])
                    except (ValueError, TypeError) as e:
                        # If conversion fails, write zeros with a note in the metric name
                        writer.writerow([
                            f"{metric_name} (invalid numeric data)",
                            0.000,
                            0.000,
                            0.000,
                            0.000
                        ])
            
            writer.writerow([])  # Empty row for separation
            
            # Write Order Status Summary section
            writer.writerow(['Order Status Summary'])
            writer.writerow(['Status', 'Count'])
            if 'order_status' in metrics:
                for status, count in metrics['order_status'].items():
                    try:
                        # Ensure count is converted to integer with proper error handling
                        count_val = int(float(count)) if count is not None else 0
                        writer.writerow([status, count_val])
                    except (ValueError, TypeError):
                        # If conversion fails, write 0 with a note
                        writer.writerow([f"{status} (invalid count)", 0])

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
                'Is On Time', 'Handler'
            ])
            
            # Write order lifecycle events
            event_index = 0
            for order in all_orders:
                # Get the complete status history
                status_history = order.get_status_history()
                
                for event in status_history:
                    # Extract event data
                    event_data = {
                        'id': order.id,
                        'event_time': event['timestamp'],
                        'status': event['status'],
                        'current_location': event['location'],
                        'handler': event.get('handler', 'unknown'),
                        'production_time': float(getattr(order, 'production_time', 0)),
                        'transit_time': float(getattr(order, 'transit_time', 0)),
                        'delay_time': float(getattr(order, 'delay_time', 0)),
                        'expected_delivery_time': order.expected_delivery_time,
                        'actual_delivery_time': order.actual_delivery_time,
                        'transportation_mode': getattr(order, 'transportation_mode', 'UNKNOWN'),
                        'source_region': order.source_region,
                        'destination_region': order.destination_region,
                        'simulation_day': event_index % 10  # Simple simulation day calculation
                    }
                    
                    # Format timestamps
                    event_time_str = event_data['event_time'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(event_data['event_time'], datetime) else str(event_data['event_time'])
                    expected_delivery_str = event_data['expected_delivery_time'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(event_data['expected_delivery_time'], datetime) else 'NA'
                    actual_delivery_str = event_data['actual_delivery_time'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(event_data['actual_delivery_time'], datetime) else 'NA'
                    
                    # Calculate derived fields
                    is_delayed = False
                    is_on_time = False
                    
                    if actual_delivery_str != 'NA':
                        # Order has been delivered - check if it was delayed
                        is_delayed = bool(event_data['delay_time'] > 0)
                        is_on_time = not is_delayed
                    elif expected_delivery_str != 'NA':
                        # Order not delivered yet - check if it's past expected delivery
                        expected_time = event_data['expected_delivery_time']
                        current_time = event_data['event_time']
                        if isinstance(expected_time, datetime) and isinstance(current_time, datetime):
                            is_delayed = current_time > expected_time
                            is_on_time = not is_delayed
                    
                    # Write row with proper type handling
                    row = [
                        event_index,
                        str(event_data['id']),
                        event_time_str,
                        str(event_data['status'].value if hasattr(event_data['status'], 'value') else event_data['status']),
                        str(event_data['current_location'].value if hasattr(event_data['current_location'], 'value') else event_data['current_location']),
                        event_data['production_time'],
                        event_data['transit_time'],
                        event_data['delay_time'],
                        expected_delivery_str,
                        actual_delivery_str,
                        str(event_data['transportation_mode'].value if hasattr(event_data['transportation_mode'], 'value') else event_data['transportation_mode']),
                        str(event_data['source_region'].value if hasattr(event_data['source_region'], 'value') else event_data['source_region']),
                        str(event_data['destination_region'].value if hasattr(event_data['destination_region'], 'value') else event_data['destination_region']),
                        event_data['simulation_day'],
                        is_delayed,
                        is_on_time,
                        event_data['handler']
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
        
        # Get all agents directly from world
        agents = getattr(world, 'agents', [])
        interactions = []
        
        # Collect all interactions
        for agent in agents:
            agent_interactions = getattr(agent, 'interactions', [])
            for interaction in agent_interactions:
                if interaction:  # Skip None or empty interactions
                    # Handle interaction data stored as dictionary
                    if isinstance(interaction, dict):
                        interactions.append({
                            'agent_id': str(getattr(agent, 'id', agent.name)),  # Use name if id not available
                            'agent_type': getattr(agent, 'agent_type', type(agent).__name__),
                            'interaction_type': str(interaction.get('type', 'unknown')),
                            'timestamp': interaction.get('timestamp', datetime.now()),
                            'target_agent': str(interaction.get('target_agent', 'unknown')),
                            'order_id': str(interaction.get('order_id', 'unknown')),
                            'status': interaction.get('status', 'unknown'),
                            'success': bool(interaction.get('success', False)),
                            'message': str(interaction.get('message', '')),
                            'simulation_day': int(interaction.get('simulation_day', 0))
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