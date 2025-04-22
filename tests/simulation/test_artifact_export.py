import unittest
from datetime import datetime, timedelta
import os
import json
from typing import Dict, Any

from models.enums import Region, OrderStatus
from models.order import Order
from simulation.world import SimulationWorld, create_simulation_world
from agents.supplier import SupplierAgent
from agents.transportation import TransportationAgent
from agents.production_facility import ProductionFacilityAgent
from agents.regional_manager import RegionalManagerAgent
from tests.test_helpers import TestArtifactGenerator
from simulation.config import DEFAULT_CONFIG
from simulation.world import simulate_supply_chain_operation

class TestArtifactExport(unittest.TestCase):
    """Test cases for verifying artifact export functionality."""
    
    def setUp(self):
        """Set up test environment with a simple supply chain network."""
        self.config = DEFAULT_CONFIG.copy()
        self.config['simulation_days'] = 10  # Ensure enough time for orders to complete
        
        # Add required configurations for agents
        self.config['supplier'] = {
            'quality': 0.9,
            'reliability': 0.9,
            'cost_efficiency': 0.8,
            'capacity': {
                'North America': 200,
                'Europe': 200,
                'East Asia': 200,
                'Southeast Asia': 200,
                'South Asia': 200
            }
        }
        self.config['transportation'] = {
            'speed': 1.0,
            'reliability': 0.9
        }
        self.config['production_facility'] = {
            'capacity': {
                'North America': 200,
                'Europe': 200,
                'East Asia': 200,
                'Southeast Asia': 200,
                'South Asia': 200
            },
            'efficiency': 0.9,
            'quality_control': 0.9,
            'flexibility': 0.8,
            'base_production_time': 3
        }
        
        # Add required simulation configuration
        self.config['simulation'] = {
            'seed': 42,
            'suppliers_per_region': 1,
            'base_production_time': 3,
            'base_transit_time': 2,
            'delay_probability': 0.1,
            'quality_threshold': 0.8
        }
        
        # Add required COO configuration
        self.config['coo'] = {
            'risk_tolerance': 0.7,
            'decision_threshold': 0.6,
            'strategy_preference': 'balanced'
        }
        
        # Add required regional manager configuration
        self.config['regional_manager'] = {
            'local_expertise': 0.8,
            'coordination_efficiency': 0.7,
            'risk_tolerance': 0.6,
            'order_batch_size': 10,
            'order_processing_interval': 1,
            'regional_demand_weights': {
                'North America': 0.3,
                'Europe': 0.3,
                'East Asia': 0.2,
                'Southeast Asia': 0.1,
                'South Asia': 0.1
            },
            'regional_production_costs': {
                'North America': 100,
                'Europe': 120,
                'East Asia': 80,
                'Southeast Asia': 90,
                'South Asia': 85
            },
            'cost_sensitivity': 0.7,
            'adaptability': 0.7
        }
        
        # Add required external events configuration
        self.config['external_events'] = {
            'weather': {
                'frequency': 0.1,
                'severity_range': [0.1, 0.5]
            },
            'geopolitical': {
                'frequency': 0.05,
                'severity_range': [0.2, 0.7]
            },
            'market': {
                'frequency': 0.15,
                'severity_range': [0.1, 0.4]
            }
        }
        
        # Create a simple world with one agent of each type
        self.world = create_simulation_world(self.config)
        
        # Generate a simulation ID
        self.simulation_id = f"test_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Add one agent of each type
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
            'production_facilities': [self.production],  # Add production facilities to state
            'agents': [self.supplier, self.transport, self.production, self.regional_manager]  # Add all agents to state
        }
        
        # Initialize agent interactions lists
        for agent in [self.supplier, self.transport, self.production, self.regional_manager]:
            if not hasattr(agent, 'interactions'):
                agent.interactions = []
        
        # Create test artifacts directory
        self.test_id = f"test_artifacts_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs("test_results", exist_ok=True)  # Create test_results directory if it doesn't exist
        self.artifact_generator = TestArtifactGenerator(simulation_id=self.test_id)
        
    def test_artifact_export(self):
        """Test that artifacts are properly exported for a simple simulation run."""
        # Create 5 test orders
        for i in range(5):
            order = Order(
                id=f"TEST_ORDER_{i}",
                product_type="TestProduct",
                creation_time=self.world.state['current_datetime'],
                source_region=Region.NORTH_AMERICA,
                destination_region=Region.NORTH_AMERICA,
                expected_delivery_time=self.world.state['current_datetime'] + timedelta(days=3),
                quantity=50,  # Add quantity for production calculations
                production_time=1.0,  # 1 day for production
                transit_time=24.0  # 24 hours for transit
            )
            self.world.state['active_orders'].append(order)
            # Add order to regional manager's pending orders
            self.regional_manager.receive_order(order)
        
        # Run simulation for enough steps to process orders
        metrics_history = []
        completed_count = 0
        max_steps = 20  # Maximum steps to prevent infinite loop
        
        print("\nRunning simulation steps:")
        for step in range(max_steps):
            metrics = simulate_supply_chain_operation(self.world, self.config)
            metrics_history.append(metrics)
            
            # Count completed orders
            completed_count = len(self.world.state['completed_orders'])
            print(f"Step {step + 1}: Completed orders = {completed_count}")
            
            # Stop if we've completed at least 3 orders
            if completed_count >= 3:
                break
        
        # Generate artifacts using the final metrics
        final_metrics = metrics_history[-1]
        self.artifact_generator.generate_artifacts(self.world, final_metrics, "artifact_export_test")
        
        # Verify artifact files exist and are not empty
        base_path = os.path.join("test_results", self.test_id)
        expected_files = [
            f"artifact_export_test_metrics.csv",
            f"artifact_export_test_order_lifecycle.csv",
            f"artifact_export_test_agent_interactions.csv"
        ]
        
        print("\nVerifying artifact files:")
        for filename in expected_files:
            filepath = os.path.join(base_path, filename)
            print(f"Checking {filename}...")
            
            # Check file exists
            self.assertTrue(os.path.exists(filepath), f"Artifact file {filename} not found")
            
            # Check file has data beyond header
            with open(filepath, 'r') as f:
                lines = f.readlines()
                self.assertGreater(len(lines), 1, 
                    f"Artifact file {filename} only contains header, no data rows found")
                
                # Print first few lines of each file for verification
                print(f"\nFirst few lines of {filename}:")
                for i, line in enumerate(lines[:5]):  # Print first 5 lines
                    print(line.strip())
        
        # Verify we have processed enough orders
        self.assertGreaterEqual(completed_count, 3, 
            f"Not enough orders completed. Expected >= 3, got {completed_count}")
        
        print(f"\nTest completed successfully. Artifacts generated in: {base_path}") 