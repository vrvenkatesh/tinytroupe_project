"""Tests for the ExternalEventAgent class."""

import pytest
from unittest.mock import patch
import random

from agents.external_event import ExternalEventAgent
from models.enums import Region

@pytest.fixture
def event_config():
    return {
        'frequency': 1.0,  # Always generate events for testing
        'severity': 1.0,
    }

@pytest.fixture
def weather_agent(event_config):
    return ExternalEventAgent(
        name="WeatherEventAgent",
        config=event_config,
        simulation_id="test_sim_001",
        event_type="weather"
    )

@pytest.fixture
def market_agent(event_config):
    return ExternalEventAgent(
        name="MarketEventAgent",
        config=event_config,
        simulation_id="test_sim_001",
        event_type="market"
    )

def test_agent_initialization():
    """Test that agent initialization works correctly."""
    config = {'frequency': 0.5, 'severity': 0.8}
    
    # Test valid initialization
    agent = ExternalEventAgent(
        name="TestAgent",
        config=config,
        simulation_id="test_sim_001",
        event_type="weather"
    )
    assert agent.name == "TestAgent"
    assert agent.event_type == "weather"
    assert agent.config == config

    # Test invalid event type
    with pytest.raises(ValueError):
        ExternalEventAgent(
            name="InvalidAgent",
            config=config,
            simulation_id="test_sim_001",
            event_type="invalid_type"
        )

def test_event_generation(weather_agent):
    """Test that event generation works correctly."""
    world_state = {}
    event = weather_agent.generate_event(world_state)
    
    assert event is not None
    assert event['type'] == "weather"
    assert isinstance(event['severity'], float)
    assert isinstance(event['duration'], int)
    assert isinstance(event['affected_regions'], list)
    assert all(isinstance(r, Region) for r in event['affected_regions'])

def test_severity_calculation(weather_agent):
    """Test that severity calculation is within expected bounds."""
    world_state = {}
    severities = [weather_agent._calculate_severity(world_state) for _ in range(100)]
    
    # Weather events should have severity factor of 0.7
    assert all(0 < s <= 1.0 for s in severities)
    assert 0.5 < sum(severities) / len(severities) < 0.9  # Average should be around 0.7

def test_duration_calculation(weather_agent, market_agent):
    """Test that duration calculation matches event type."""
    world_state = {}
    
    # Test weather event durations (base: 7 days)
    weather_durations = [weather_agent._calculate_duration(world_state) for _ in range(100)]
    assert all(3 <= d <= 10 for d in weather_durations)
    
    # Test market event durations (base: 90 days)
    market_durations = [market_agent._calculate_duration(world_state) for _ in range(100)]
    assert all(45 <= d <= 135 for d in market_durations)

def test_affected_regions(weather_agent, market_agent):
    """Test that affected regions are determined correctly based on event type."""
    world_state = {}
    
    # Weather events should affect exactly one region
    weather_regions = weather_agent._determine_affected_regions(world_state)
    assert len(weather_regions) == 1
    assert isinstance(weather_regions[0], Region)
    
    # Market events should affect all regions
    market_regions = market_agent._determine_affected_regions(world_state)
    assert len(market_regions) == len(list(Region))
    assert set(market_regions) == set(Region)

def test_custom_act(weather_agent):
    """Test the custom_act method with different return types."""
    world_state = {}
    
    # Test without return_actions
    result = weather_agent.custom_act(world_state)
    assert isinstance(result, str)
    assert "Generated weather event" in result
    
    # Test with return_actions
    result_list = weather_agent.custom_act(world_state, return_actions=True)
    assert isinstance(result_list, list)
    assert len(result_list) == 1
    assert isinstance(result_list[0], str)

def test_error_handling(weather_agent):
    """Test error handling in custom_act."""
    with patch.object(weather_agent, 'generate_event', side_effect=Exception("Test error")):
        result = weather_agent.custom_act()
        assert "Error in External Event Generator" in result
        assert "Test error" in result 