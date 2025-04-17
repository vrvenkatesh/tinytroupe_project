"""
Test suite for supply chain simulation components.
"""

import unittest
import random
from typing import Dict, Any
import uuid
from datetime import datetime, timedelta

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
    create_logistics_provider_agent,
    simulate_supply_chain_operation,
    OrderStatus,
    Order,
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
        agent_name = f"COO_{self.simulation_id}"
        coo = create_coo_agent(agent_name, self.config['coo'], self.simulation_id)
        self.assertIsInstance(coo, TinyPerson)
        
        # Test persona definition
        self.assertEqual(coo._persona['occupation']['title'], 'Chief Operating Officer')
        self.assertEqual(coo._persona['occupation']['organization'], 'Global Supply Chain Corp')

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
            manager._persona['occupation']['title'],
            'Regional Manager'
        )
        
        # Test decision making parameters
        decision_making = manager._persona['decision_making']
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
        capabilities = supplier._persona['capabilities']
        self.assertIn('production_capacity', capabilities)
        self.assertIn('reliability', capabilities)
        self.assertIn('lead_time', capabilities)
        self.assertIn('quality_score', capabilities)
        
        # Test region assignment
        self.assertIn('region', supplier._persona['occupation'])
        self.assertIsInstance(supplier._persona['occupation']['region'], Region)
        self.assertIn(supplier._persona['occupation']['region'], list(Region))

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
        
        logistics_providers = {
            "TestLogistics": create_logistics_provider_agent(
                source_region=Region.NORTH_AMERICA,
                dest_region=Region.EUROPE,
                config=self.config['logistics'],
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
        self.world.state.update({
            'coo_agent': coo,
            'regional_managers': list(regional_managers.values()),
            'suppliers': [s for suppliers_list in suppliers.values() for s in suppliers_list],
            'production_facilities': list(production_facilities.values()),
            'logistics_providers': list(logistics_providers.values()),
            'active_orders': [],
            'completed_orders': [],
            'order_lifecycle': {}
        })
        
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

    def test_agent_interactions(self):
        """Test basic agent interactions."""
        # Create agents with unique names
        coo = create_coo_agent(f"COO_{self.simulation_id}", self.config['coo'], self.simulation_id)
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
        manager.listen(f"What is the current status in {manager._persona['occupation']['region']}?")
        response = manager.act()
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
        
        # Test supplier interaction
        supplier.listen("Update your production and delivery status")
        response = supplier.act()
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_basic_order_flow(self):
        """Test the basic flow of an order through all states."""
        # Create minimal set of agents needed
        region = Region.NORTH_AMERICA
    
        # Create one of each type of agent
        manager = create_regional_manager_agent(
            f"TestManager_{self.simulation_id}",
            {**self.config['regional_manager'], 'region': region},
            self.simulation_id
        )
    
        supplier = create_supplier_agent(
            f"TestSupplier_{self.simulation_id}",
            {**self.config['supplier'], 'region': region},
            self.simulation_id
        )
    
        facility = create_production_facility_agent(
            name=f"Facility_{region.value}",
            config={
                'region': region,
                'capacity': 100,
                'efficiency': 0.9,
                'quality_rate': 1.0,  # Set to 1.0 for tests to ensure quality checks pass
                'flexibility': 0.8,
                'base_production_time': 2
            },
            simulation_id=self.simulation_id
        )
    
        logistics = create_logistics_provider_agent(
            source_region=Region.NORTH_AMERICA,
            dest_region=Region.EUROPE,
            config=self.config['logistics']
        )
    
        # Add agents to world
        self.world.add_agent(manager)
        self.world.add_agent(supplier)
        self.world.add_agent(facility)
        self.world.add_agent(logistics)
    
        # Initialize world state with all required fields
        self.world.state.update({
            'regional_managers': [manager],
            'suppliers': [supplier],
            'production_facilities': [facility],
            'logistics_providers': [logistics],
            'active_orders': [],
            'completed_orders': [],
            'failed_orders': [],
            'current_datetime': datetime.now(),
            'config': self.config,
            'risk_exposure': 0.5,
            'cost_pressure': 0.5,
            'demand_volatility': 0.5,
            'supply_risk': 0.5,
            'reliability_requirement': 0.5,
            'flexibility_requirement': 0.5,
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
    
        # Create a test order in NEW state
        test_order = Order(
            id='TEST_ORDER_1',
            product_type='Standard',
            quantity=10,
            source_region=region,
            destination_region=Region.EUROPE,
            creation_time=datetime.now(),
            expected_delivery_time=datetime.now() + timedelta(days=2),
            status=OrderStatus.NEW,
            current_location=region
        )
    
        # Add order to world state
        self.world.state['active_orders'] = [test_order]
    
        # Run one step - Regional Manager should assign order to facility and supplier
        metrics = simulate_supply_chain_operation(self.world, self.config)
    
        # Verify order was assigned and moved to PRODUCTION
        updated_order = next(order for order in self.world.state['active_orders'] if order.id == 'TEST_ORDER_1')
        self.assertEqual(updated_order.status, OrderStatus.PRODUCTION)
        self.assertIsNotNone(updated_order.production_facility)
        self.assertIsNotNone(getattr(updated_order, 'supplier', None))
    
        # Run another step - Production should progress
        metrics = simulate_supply_chain_operation(self.world, self.config)
        updated_order = next(order for order in self.world.state['active_orders'] if order.id == 'TEST_ORDER_1')
        self.assertTrue(updated_order.production_time > 0)
    
        # Run until production is complete
        while updated_order.status != OrderStatus.READY_FOR_SHIPPING:
            metrics = simulate_supply_chain_operation(self.world, self.config)
            updated_order = next(order for order in self.world.state['active_orders'] if order.id == 'TEST_ORDER_1')
    
        # Verify order is ready for shipping
        self.assertEqual(updated_order.status, OrderStatus.READY_FOR_SHIPPING)
    
        # Run until shipping starts
        while updated_order.status != OrderStatus.IN_TRANSIT:
            metrics = simulate_supply_chain_operation(self.world, self.config)
            updated_order = next(order for order in self.world.state['active_orders'] if order.id == 'TEST_ORDER_1')
    
        # Verify order is in transit
        self.assertEqual(updated_order.status, OrderStatus.IN_TRANSIT)
        self.assertIsNotNone(updated_order.transportation_mode)
    
        # Run until delivery
        while updated_order.status != OrderStatus.DELIVERED:
            metrics = simulate_supply_chain_operation(self.world, self.config)
            try:
                updated_order = next(order for order in self.world.state['active_orders'] if order.id == 'TEST_ORDER_1')
            except StopIteration:
                # Order might have moved to completed_orders
                updated_order = next(order for order in self.world.state['completed_orders'] if order.id == 'TEST_ORDER_1')
    
        # Verify order was delivered
        self.assertEqual(updated_order.status, OrderStatus.DELIVERED)
        self.assertIsNotNone(updated_order.actual_delivery_time)

if __name__ == '__main__':
    unittest.main() 