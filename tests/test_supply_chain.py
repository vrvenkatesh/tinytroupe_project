"""
Test suite for supply chain simulation components.
"""

import unittest
import random
from typing import Dict, Any

from tinytroupe.agent import TinyPerson
from tinytroupe.environment import TinyWorld

from supply_chain import (
    DEFAULT_CONFIG,
    Region,
    create_coo_agent,
    create_regional_manager_agent,
    create_supplier_agent,
    create_simulation_world,
)

class TestSupplyChainComponents(unittest.TestCase):
    """Test cases for supply chain components."""

    def setUp(self):
        """Set up test fixtures."""
        random.seed(42)  # For reproducibility
        self.config = DEFAULT_CONFIG
        self.world = create_simulation_world(self.config)
        self.simulation_id = "test_simulation"

    def test_world_creation(self):
        """Test creation of simulation world."""
        self.assertIsInstance(self.world, TinyWorld)
        self.assertEqual(self.world.name, "Global Supply Network")
        self.assertIn('risk_exposure', self.world.state)
        self.assertIn('cost_pressure', self.world.state)
        
        # Test regional state initialization
        for region in Region:
            self.assertIn(f'{region.value}_risk', self.world.state)
            self.assertIn(f'{region.value}_cost', self.world.state)
            self.assertIn(f'{region.value}_demand', self.world.state)

    def test_coo_agent_creation(self):
        """Test creation of COO agent."""
        coo = create_coo_agent("TestCOO", self.config['coo'], self.simulation_id)
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
        manager = create_regional_manager_agent(
            "TestManager",
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
        supplier = create_supplier_agent(
            "TestSupplier",
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

    def test_agent_interactions(self):
        """Test basic agent interactions."""
        # Create agents
        coo = create_coo_agent("TestCOO", self.config['coo'], self.simulation_id)
        manager = create_regional_manager_agent(
            "TestManager",
            self.config['regional_manager'],
            self.simulation_id
        )
        supplier = create_supplier_agent(
            "TestSupplier",
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