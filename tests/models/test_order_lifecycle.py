import unittest
from datetime import datetime, timedelta
import pandas as pd
import sys

from models.enums import Region, OrderStatus
from models.order import Order
from simulation.world import SimulationWorld, create_simulation_world, simulate_supply_chain_operation
from agents.supplier import SupplierAgent
from agents.transportation import TransportationAgent
from agents.production_facility import ProductionFacilityAgent
from agents.regional_manager import RegionalManagerAgent
from simulation.config import DEFAULT_CONFIG

class TestOrderLifecycle(unittest.TestCase):
    """Test cases for verifying order lifecycle tracking."""
    
    def setUp(self):
        """Set up test environment with minimal supply chain network."""
        self.config = DEFAULT_CONFIG.copy()
        self.config['simulation_days'] = 10
        
        # Add required configurations for agents (simplified from test_artifact_export)
        self.config['supplier'] = {
            'quality': 0.9,
            'reliability': 0.9,
            'cost_efficiency': 0.8,
            'capacity': {region.value: 200 for region in Region}
        }
        
        self.config['transportation'] = {
            'speed': 1.0,
            'reliability': 0.9
        }
        
        self.config['production_facility'] = {
            'capacity': {region.value: 200 for region in Region},
            'efficiency': 0.9,
            'quality_control': 0.9,
            'flexibility': 0.8,
            'base_production_time': 3
        }
        
        self.config['simulation'] = {
            'seed': 42,
            'suppliers_per_region': 1,
            'base_production_time': 3,
            'base_transit_time': 2,
            'delay_probability': 0.1,
            'quality_threshold': 0.8
        }
        
        self.config['regional_manager'] = {
            'local_expertise': 0.8,
            'coordination_efficiency': 0.7,
            'risk_tolerance': 0.6,
            'order_batch_size': 10,
            'order_processing_interval': 1,
            'regional_demand_weights': {
                region.value: 0.2 for region in Region
            },
            'regional_production_costs': {
                region.value: 100 for region in Region
            },
            'cost_sensitivity': 0.7,
            'adaptability': 0.7
        }
        
        # Create simulation world
        self.world = create_simulation_world(self.config)
        
        # Generate simulation ID
        self.simulation_id = f"test_lifecycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Add one agent of each type in North America
        self.supplier = SupplierAgent(
            name="TestSupplier",
            config=self.config,
            simulation_id=self.simulation_id,
            region=Region.NORTH_AMERICA,
            supplier_type="tier1"
        )
        self.transport = TransportationAgent(
            name="TestTransport",
            config=self.config,
            simulation_id=self.simulation_id
        )
        self.production = ProductionFacilityAgent(
            name="TestProduction",
            config=self.config,
            simulation_id=self.simulation_id,
            region=Region.NORTH_AMERICA
        )
        
        # Add regional manager for North America
        self.regional_manager = RegionalManagerAgent(
            name="TestRegionalManager",
            config=self.config,
            simulation_id=self.simulation_id,
            region=Region.NORTH_AMERICA
        )
        
        self.world.add_agent(self.supplier)
        self.world.add_agent(self.transport)
        self.world.add_agent(self.production)
        self.world.add_agent(self.regional_manager)  # Add regional manager to world
        
        # Initialize world state
        self.world.state = {
            'active_orders': [],
            'completed_orders': [],
            'current_datetime': datetime.now(),
            'risk_levels': {
                'supply_risk': 0.3,
                'demand_risk': 0.3,
                'operational_risk': 0.3
            },
            'metrics': {
                'resilience_score': 0.7,
                'recovery_time': timedelta(),
                'risk_exposure_trend': []
            },
            'production_facilities': [self.production]  # Add production facilities to state
        }
        
        # Initialize list to track all order events
        self.expected_events = []
    
    def test_order_lifecycle_events(self):
        """Test that order lifecycle events are properly tracked."""
        # Create 5 test orders and track creation events
        num_orders = 5
        for i in range(num_orders):
            order = Order(
                id=f"TEST_ORDER_{i}",
                product_type="TestProduct",
                creation_time=self.world.state['current_datetime'],
                source_region=Region.NORTH_AMERICA,
                destination_region=Region.NORTH_AMERICA,
                expected_delivery_time=self.world.state['current_datetime'] + timedelta(days=3),
                quantity=50
            )
            self.world.state['active_orders'].append(order)
            
            # Track order creation event
            self.expected_events.append({
                'order_id': order.id,
                'event_type': 'ORDER_CREATED',
                'timestamp': order.creation_time,
                'status': OrderStatus.NEW.value
            })
        
        # Run simulation and track status changes
        metrics_history = []
        completed_count = 0
        max_steps = 20
        
        # Force output to stdout
        print("\nRunning simulation steps and tracking events:", file=sys.stdout, flush=True)
        for step in range(max_steps):
            metrics = simulate_supply_chain_operation(self.world, self.config)
            metrics_history.append(metrics)
            
            # Track status changes for all orders
            for order in self.world.state['active_orders'] + self.world.state['completed_orders']:
                # Get status history using the public method
                status_history = order.get_status_history()
                for status_change in status_history:
                    if not any(e['order_id'] == order.id and 
                             e['timestamp'] == status_change['timestamp'] and
                             e['status'] == status_change['status'].value
                             for e in self.expected_events):
                        self.expected_events.append({
                            'order_id': order.id,
                            'event_type': 'STATUS_CHANGED',
                            'timestamp': status_change['timestamp'],
                            'status': status_change['status'].value,
                            'updated_by': status_change.get('handler', 'system')
                        })
            
            # Count completed orders
            completed_count = len(self.world.state['completed_orders'])
            print(f"Step {step + 1}: Completed orders = {completed_count}, "
                  f"Total events tracked = {len(self.expected_events)}", 
                  file=sys.stdout, flush=True)
            
            # Stop if we've completed all orders or reached max steps
            if completed_count >= num_orders:
                break
        
        # Convert events to DataFrame for analysis
        events_df = pd.DataFrame(self.expected_events)
        
        # Basic assertions
        self.assertGreater(len(events_df), num_orders, 
            "Should have more events than initial orders")
        
        # Verify each order has a creation event
        creation_events = events_df[events_df['event_type'] == 'ORDER_CREATED']
        self.assertEqual(len(creation_events), num_orders,
            f"Expected {num_orders} creation events, got {len(creation_events)}")
        
        # Verify that not all updated_by fields are NaN
        non_nan_handlers = events_df['updated_by'].notna().sum()
        self.assertGreater(non_nan_handlers, 0,
            "No events have handler information. Expected some events to have non-null handlers.")
        print(f"\nEvents with handler information: {non_nan_handlers} out of {len(events_df)}", 
              file=sys.stdout, flush=True)
        
        # Verify that orders are progressing beyond NEW status
        status_counts = events_df['status'].value_counts()
        non_new_statuses = status_counts[status_counts.index != OrderStatus.NEW.value].sum()
        self.assertGreater(non_new_statuses, 0,
            f"No orders progressed beyond NEW status. Status counts:\n{status_counts}")
        print("\nStatus distribution:", file=sys.stdout, flush=True)
        print(status_counts, file=sys.stdout, flush=True)
        
        # Verify each completed order has a DELIVERED status
        if completed_count > 0:
            delivered_orders = events_df[
                (events_df['event_type'] == 'STATUS_CHANGED') & 
                (events_df['status'] == OrderStatus.DELIVERED.value)
            ]
            self.assertEqual(len(delivered_orders), completed_count,
                f"Expected {completed_count} DELIVERED events, got {len(delivered_orders)}")
        
        # Verify status transition sequence for each order
        expected_status_sequence = [
            OrderStatus.NEW.value,
            OrderStatus.PRODUCTION.value,
            OrderStatus.READY_FOR_SHIPPING.value,
            OrderStatus.IN_TRANSIT.value,
            OrderStatus.DELIVERED.value
        ]
        
        for order_id in events_df['order_id'].unique():
            order_events = events_df[events_df['order_id'] == order_id].sort_values('timestamp')
            status_sequence = order_events[order_events['event_type'] == 'STATUS_CHANGED']['status'].tolist()
            
            # Check that each status change follows the expected sequence
            current_status_idx = 0
            for status in status_sequence:
                while (current_status_idx < len(expected_status_sequence) and 
                       status != expected_status_sequence[current_status_idx]):
                    current_status_idx += 1
                self.assertLess(current_status_idx, len(expected_status_sequence),
                    f"Unexpected status {status} for order {order_id}")
        
        print("\nTest completed successfully.", file=sys.stdout, flush=True)
        print(f"Total events tracked: {len(events_df)}", file=sys.stdout, flush=True)
        print(f"Events per order: {len(events_df) / num_orders:.2f}", file=sys.stdout, flush=True)
        print("\nSample of events dataframe:", file=sys.stdout, flush=True)
        print(events_df.head(), file=sys.stdout, flush=True) 