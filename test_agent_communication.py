import unittest
import uuid
from datetime import datetime, timedelta

from tinytroupe.environment.tiny_world import TinyWorld
from tinytroupe.agent.tiny_person import TinyPerson

class TestAgentCommunication(unittest.TestCase):
    """Test cases for verifying basic agent communication."""

    def setUp(self):
        """Set up test environment."""
        # Configure logging
        import logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create world with unique name
        self.world_id = str(uuid.uuid4())[:8]
        self.world = TinyWorld(f"TestWorld_{self.world_id}")
        
        # Create test agents
        self.agent1 = TinyPerson(f"Agent1_{self.world_id}")
        self.agent2 = TinyPerson(f"Agent2_{self.world_id}")
        
        # Add agents to world
        self.world.add_agent(self.agent1)
        self.world.add_agent(self.agent2)
        
        # Make agents accessible to each other
        self.agent1.make_agent_accessible(self.agent2)
        self.agent2.make_agent_accessible(self.agent1)

    def test_basic_communication(self):
        """Test basic communication between agents."""
        # Agent1 thinks about sending a message
        message = "Hello Agent2, how are you?"
        self.agent1.think(message)
        
        # Verify the message was recorded
        memories = self.agent1.retrieve_recent_memories()
        self.assertTrue(any(message in str(m) for m in memories))

    def test_action_communication(self):
        """Test communication of actions between agents."""
        # Agent1 performs an action
        action_content = {
            "action": {
                "type": "TEST_ACTION",
                "description": "test_action",
                "target": self.agent2.name
            },
            "cognitive_state": {
                "goals": ["Complete test action"],
                "attention": ["Focused on test"],
                "emotions": ["Neutral"]
            }
        }
        
        # Store action in memory
        self.agent1.store_in_memory({
            'role': 'assistant',
            'content': action_content,
            'type': 'action',
            'simulation_timestamp': self.agent1.iso_datetime()
        })
        
        # Display the action
        self.agent1._display_communication(
            role='assistant',
            content=action_content,
            kind='action',
            simplified=True
        )
        
        # Verify the action was recorded
        memories = self.agent1.retrieve_recent_memories()
        self.assertTrue(any("test_action" in str(m) for m in memories))

    def tearDown(self):
        """Clean up after each test."""
        # Clean up agents
        self.agent1.make_all_agents_inaccessible()
        self.agent2.make_all_agents_inaccessible()
        TinyPerson.clear_agents()

if __name__ == '__main__':
    unittest.main() 