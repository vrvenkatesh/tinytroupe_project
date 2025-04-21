import unittest
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List

from tinytroupe.agent.tiny_person import TinyPerson
from tinytroupe.environment.tiny_world import TinyWorld
from supply_chain import (
    Region, OrderStatus, Order,
    create_regional_manager_agent,
    create_production_facility_agent,
    create_simulation_world
)

class TestAgentInteractions(unittest.TestCase):
    """Test cases for verifying agent interactions in supply chain simulation."""

    def setUp(self):
        """Set up test environment with minimal configuration."""
        # Configure logging
        import logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Generate unique simulation ID
        self.simulation_id = str(uuid.uuid4())[:8]
        
        # Create minimal test configuration
        self.config = {
            'simulation': {
                'time_steps': 5,
                'base_demand': 5,
                'monte_carlo_iterations': 1,
                'suppliers_per_region': 1
            },
            'regional_manager': {
                'local_expertise': 0.8,
                'adaptability': 0.7,
                'cost_sensitivity': 0.6,
                'capabilities': {
                    'decision_speed': 0.8,
                    'coordination': 0.7
                }
            },
            'production_facility': {
                'capacity': 100,
                'efficiency': 0.8,
                'quality_score': 0.9,
                'capabilities': {
                    'production_rate': 10,
                    'quality_control': 0.9
                }
            }
        }
        
        # Create simulation world
        self.world = TinyWorld(
            name=f"TestWorld_{self.simulation_id}",
            agents=[],
            broadcast_if_no_target=True
        )
        
        # Initialize world state
        self.world.state = {
            'active_orders': [],
            'completed_orders': [],
            'failed_orders': [],
            'current_datetime': datetime.now(),
            'config': self.config
        }

    def tearDown(self):
        """Clean up after each test."""
        # Clear all agents
        self.world.agents = []
        self.world.state['active_orders'] = []
        self.world.state['completed_orders'] = []
        self.world.state['failed_orders'] = []

    def test_order_assignment_interaction(self):
        """Test interaction between Regional Manager and Production Facility during order assignment."""
        # Create test agents
        manager = create_regional_manager_agent(
            name=f"Manager_NA_{self.simulation_id}",
            config={**self.config['regional_manager'], 'region': Region.NORTH_AMERICA},
            simulation_id=self.simulation_id
        )
        
        facility = create_production_facility_agent(
            name=f"Facility_NA_{self.simulation_id}",
            config={**self.config['production_facility'], 'region': Region.NORTH_AMERICA},
            simulation_id=self.simulation_id
        )
        
        # Add agents to world
        self.world.add_agent(manager)
        self.world.add_agent(facility)
        
        # Update world state
        self.world.state['regional_managers'] = [manager]
        self.world.state['production_facilities'] = [facility]
        
        # Create test order
        test_order = Order(
            id=f"TEST_ORDER_{self.simulation_id}",
            product_type="Standard",
            quantity=10,
            source_region=Region.NORTH_AMERICA,
            destination_region=Region.NORTH_AMERICA,
            creation_time=self.world.state['current_datetime'],
            expected_delivery_time=self.world.state['current_datetime'] + timedelta(days=2)
        )
        self.world.state['active_orders'].append(test_order)
        
        # Enable display of communications for both agents
        manager.communication_display = True
        facility.communication_display = True
        
        # Run simulation steps
        for step in range(self.config['simulation']['time_steps']):
            self.logger.info(f"\nSimulation step {step + 1}")
            
            # Manager processes orders
            manager_response = manager.act(self.world.state)
            self.logger.info(f"Manager response: {manager_response}")
            
            # Facility processes assigned orders
            facility_response = facility.act(self.world.state)
            self.logger.info(f"Facility response: {facility_response}")
            
            # Verify interactions were recorded
            manager_interactions = manager.pretty_current_interactions()
            facility_interactions = facility.pretty_current_interactions()
            
            self.logger.info(f"Manager interactions: {manager_interactions}")
            self.logger.info(f"Facility interactions: {facility_interactions}")
            
            # Assert interactions were recorded
            if step == 0:  # First step should have order assignment interaction
                self.assertIn("Assigned order", str(manager_interactions))
            
            # Update simulation time
            self.world.state['current_datetime'] = datetime.now()

if __name__ == '__main__':
    unittest.main() 