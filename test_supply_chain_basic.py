import unittest
from datetime import datetime
from supply_chain import (
    create_simulation_world,
    simulate_supply_chain_operation,
    Region,
    OrderStatus,
    DEFAULT_CONFIG
)

class TestSupplyChainBasic(unittest.TestCase):
    def setUp(self):
        self.config = DEFAULT_CONFIG.copy()
        # Use smaller values for testing
        self.config['simulation'].update({
            'monte_carlo_iterations': 2,
            'suppliers_per_region': 2,
            'time_steps': 5,
            'base_demand': 5
        })
        self.world = create_simulation_world(self.config)

    def test_agent_initialization(self):
        """Test that all necessary agents are created and added to the world."""
        # Count agents by type
        regional_managers = [agent for agent in self.world.agents 
                           if 'Manager' in agent.name and any(region.value in agent.name for region in Region)]
        suppliers = [agent for agent in self.world.agents 
                    if 'Supplier' in agent.name]
        production_facilities = [agent for agent in self.world.agents 
                               if 'Facility' in agent.name]
        logistics_providers = [agent for agent in self.world.agents 
                             if 'Logistics' in agent.name]

        # Print agent counts for debugging
        print(f"\nAgent counts:")
        print(f"Regional Managers: {len(regional_managers)}")
        print(f"Suppliers: {len(suppliers)}")
        print(f"Production Facilities: {len(production_facilities)}")
        print(f"Logistics Providers: {len(logistics_providers)}")

        # Verify counts
        self.assertEqual(len(regional_managers), len(Region), "Should have one manager per region")
        self.assertEqual(len(suppliers), len(Region) * self.config['simulation']['suppliers_per_region'],
                        "Should have correct number of suppliers")
        self.assertEqual(len(production_facilities), len(Region),
                        "Should have one production facility per region")
        self.assertTrue(len(logistics_providers) > 0, "Should have logistics providers")

    def test_single_simulation_step(self):
        """Test a single simulation step to verify order processing."""
        # Run one simulation step
        metrics = simulate_supply_chain_operation(self.world, self.config)

        # Verify that metrics were calculated
        self.assertIsInstance(metrics, dict, "Should return metrics dictionary")
        self.assertTrue('order_status' in metrics, "Should include order status")
        
        # Print detailed metrics for debugging
        print("\nSimulation Step Metrics:")
        for key, value in metrics.items():
            if key != 'order_status':
                print(f"{key}: {value}")
            else:
                print("\nOrder Status Counts:")
                for status, count in value.items():
                    print(f"{status}: {count}")

    def test_order_processing(self):
        """Test that orders move through different statuses."""
        # Run multiple simulation steps
        for _ in range(3):  # Run 3 steps to allow orders to progress
            metrics = simulate_supply_chain_operation(self.world, self.config)
            
            # Get all orders
            active_orders = self.world.state.get('active_orders', [])
            completed_orders = self.world.state.get('completed_orders', [])
            
            print(f"\nStep {_+1} Order Status:")
            status_counts = {status.value: 0 for status in OrderStatus}
            for order in active_orders + completed_orders:
                status_counts[order.status.value] += 1
            
            for status, count in status_counts.items():
                print(f"{status}: {count}")

if __name__ == '__main__':
    unittest.main() 