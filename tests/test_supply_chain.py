"""
Test suite for supply chain simulation components.
"""

import unittest
import random
from typing import Dict, Any
import uuid

from tinytroupe.agent import TinyPerson
from tinytroupe.environment import TinyWorld

from supply_chain import (
    DEFAULT_CONFIG,
    Region,
    create_coo_agent,
    create_regional_manager_agent,
    create_supplier_agent,
    create_simulation_world,
    create_production_facility_agent,
    create_logistics_agent,
    simulate_supply_chain_operation,
    OrderStatus,
)

class TestSupplyChainComponents(unittest.TestCase):
    """Test cases for supply chain components."""

    def setUp(self):
        """Set up test fixtures."""
        self.simulation_id = str(uuid.uuid4())[:8]
        self.config = DEFAULT_CONFIG.copy()
        
        # Simplify configuration for testing
        self.config['simulation'].update({
            'monte_carlo_iterations': 2,
            'suppliers_per_region': 2,
            'time_steps': 5,
            'base_demand': 5,
            'regions': ['NORTH_AMERICA', 'EUROPE']
        })
        
        self.config['production_facility'].update({
            'base_production_time': 2
        })
        
        # Create test world with unique name
        self.world = create_simulation_world(self.config)
        
        # Ensure world state is properly initialized
        if 'risk_exposure' not in self.world.state:
            self.world.state.update({
                'risk_exposure': 0.5,
                'cost_pressure': 0.5,
                'demand_volatility': 0.5,
                'supply_risk': 0.5,
                'reliability_requirement': 0.5,
                'flexibility_requirement': 0.5,
                'active_orders': [],
                'completed_orders': [],
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
            })
        
        self.world.current_time = 0

    def test_world_creation(self):
        """Test creation of simulation world."""
        self.assertIsInstance(self.world, TinyWorld)
        self.assertTrue(self.world.name.startswith("SupplyChainWorld_"), f"World name '{self.world.name}' should start with 'SupplyChainWorld_'")
        self.assertIn('risk_exposure', self.world.state)
        self.assertIn('cost_pressure', self.world.state)
        
        # Test regional metrics initialization
        self.assertIn('regional_metrics', self.world.state)
        for region in Region:
            self.assertIn(region.value, self.world.state['regional_metrics'])
            metrics = self.world.state['regional_metrics'][region.value]
            self.assertIn('risk', metrics)
            self.assertIn('cost', metrics)
            self.assertIn('demand', metrics)
            self.assertIn('supply_risk', metrics)
            self.assertIn('infrastructure', metrics)
            self.assertIn('congestion', metrics)
            self.assertIn('efficiency', metrics)
            self.assertIn('flexibility', metrics)
            self.assertIn('quality', metrics)

    def test_coo_agent_creation(self):
        """Test creation of COO agent."""
        agent_name = f"TestCOO_{self.simulation_id}"
        coo = create_coo_agent(agent_name, self.config['coo'], self.simulation_id)
        self.assertIsInstance(coo, TinyPerson)
        
        # Test persona definition
        self.assertEqual(coo.persona['occupation']['title'], 'COO')
        self.assertEqual(coo.persona['occupation']['organization'], 'Tekron Industries')
        
        # Test behaviors
        behaviors = coo.persona['behaviors']
        self.assertTrue(any('metrics' in b.lower() for b in behaviors))
        self.assertTrue(any('supplier' in b.lower() for b in behaviors))

    def test_regional_manager_creation(self):
        """Test creation of regional manager agent."""
        agent_name = f"TestManager_{self.simulation_id}"
        manager = create_regional_manager_agent(
            agent_name,
            self.config['regional_manager'],
            self.simulation_id
        )
        self.assertIsInstance(manager, TinyPerson)
        
        # Test persona definition
        self.assertEqual(
            manager.persona['occupation']['title'],
            'Regional Supply Chain Manager'
        )
        
        # Test decision making parameters
        decision_making = manager.persona['decision_making']
        self.assertIn('local_expertise', decision_making)
        self.assertIn('adaptability', decision_making)
        self.assertIn('cost_sensitivity', decision_making)

    def test_supplier_agent_creation(self):
        """Test creation of supplier agent."""
        agent_name = f"TestSupplier_{self.simulation_id}"
        supplier = create_supplier_agent(
            agent_name,
            self.config['supplier'],
            self.simulation_id
        )
        self.assertIsInstance(supplier, TinyPerson)
        
        # Test capabilities
        capabilities = supplier.persona['capabilities']
        self.assertIn('quality_score', capabilities)
        self.assertIn('reliability', capabilities)
        
        # Test region assignment
        self.assertIn('region', supplier.persona)
        self.assertIn(supplier.persona['region'], 
                     [r.value for r in Region])

    def test_order_based_simulation(self):
        """Test the order-based supply chain simulation."""
        print("\n=== Testing Order-Based Simulation ===")
        
        # Create agents with unique names
        coo = create_coo_agent(f"TestCOO_{self.simulation_id}", self.config['coo'], self.simulation_id)
        
        regional_managers = {
            Region.NORTH_AMERICA: create_regional_manager_agent(
                f"TestManager_NA_{self.simulation_id}", self.config['regional_manager'], self.simulation_id
            ),
            Region.EUROPE: create_regional_manager_agent(
                f"TestManager_EU_{self.simulation_id}", self.config['regional_manager'], self.simulation_id
            )
        }
        
        suppliers = {
            Region.NORTH_AMERICA: [create_supplier_agent(
                f"TestSupplier_NA_{i}_{self.simulation_id}", self.config['supplier'], self.simulation_id
            ) for i in range(self.config['simulation']['suppliers_per_region'])],
            Region.EUROPE: [create_supplier_agent(
                f"TestSupplier_EU_{i}_{self.simulation_id}", self.config['supplier'], self.simulation_id
            ) for i in range(self.config['simulation']['suppliers_per_region'])]
        }
        
        production_facilities = {
            Region.NORTH_AMERICA: create_production_facility_agent(
                f"TestFacility_NA_{self.simulation_id}", self.config['production_facility'], self.simulation_id
            ),
            Region.EUROPE: create_production_facility_agent(
                f"TestFacility_EU_{self.simulation_id}", self.config['production_facility'], self.simulation_id
            )
        }
        
        logistics_providers = {
            "TestLogistics": create_logistics_agent(
                f"TestLogistics_{self.simulation_id}", self.config['logistics'], self.simulation_id
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
                if order.status == OrderStatus.IN_PRODUCTION:
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
            
            print("\nMetrics:", {k: f"{v:.2f}" for k, v in metrics.items()})
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

    def test_agent_interactions(self):
        """Test basic agent interactions."""
        # Create agents with unique names
        coo = create_coo_agent(f"TestCOO_{self.simulation_id}", self.config['coo'], self.simulation_id)
        manager = create_regional_manager_agent(
            f"TestManager_{self.simulation_id}",
            self.config['regional_manager'],
            self.simulation_id
        )
        supplier = create_supplier_agent(
            f"TestSupplier_{self.simulation_id}",
            self.config['supplier'],
            self.simulation_id
        )
        
        # Add agents to world
        self.world.add_agent(coo)
        self.world.add_agent(manager)
        self.world.add_agent(supplier)
        
        # Test COO interaction
        coo.listen("What is our current supply chain resilience status?")
        response = coo.act()
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
        
        # Test manager interaction
        manager.listen(f"What is the current status in {manager.persona['region']}?")
        response = manager.act()
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
        
        # Test supplier interaction
        supplier.listen("Update your production and delivery status")
        response = supplier.act()
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

if __name__ == '__main__':
    unittest.main() 