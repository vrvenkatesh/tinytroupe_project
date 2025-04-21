"""Tests for the COO agent."""

import pytest
from datetime import datetime
import uuid

from agents.coo import COOAgent
from models.enums import Region
from simulation.config import DEFAULT_CONFIG

@pytest.fixture
def coo_agent():
    """Create a COO agent for testing."""
    return COOAgent(
        name="TestCOO",
        config=DEFAULT_CONFIG['coo'],
        simulation_id=str(uuid.uuid4())
    )

@pytest.fixture
def sample_world_state():
    """Create a sample world state for testing."""
    return {
        'risk_exposure': 0.5,
        'cost_pressure': 0.4,
        'demand_volatility': 0.6,
        'supply_risk': 0.3,
        'reliability_requirement': 0.7,
        'flexibility_requirement': 0.8,
        'current_datetime': datetime.now(),
        'regional_metrics': {
            'NORTH_AMERICA': {
                'risk': 0.4,
                'cost': 0.5,
                'demand': 0.7,
                'supply_risk': 0.3,
                'infrastructure': 0.8,
                'congestion': 0.4,
                'efficiency': 0.7,
                'flexibility': 0.6,
                'quality': 0.9
            },
            'EUROPE': {
                'risk': 0.5,
                'cost': 0.6,
                'demand': 0.5,
                'supply_risk': 0.4,
                'infrastructure': 0.7,
                'congestion': 0.5,
                'efficiency': 0.6,
                'flexibility': 0.7,
                'quality': 0.8
            }
        }
    }

def test_coo_initialization(coo_agent):
    """Test COO agent initialization."""
    assert coo_agent.name == "TestCOO"
    assert coo_agent.config == DEFAULT_CONFIG['coo']
    assert isinstance(coo_agent.simulation_id, str)
    
    # Test persona definition
    assert coo_agent._persona['occupation']['title'] == 'Chief Operating Officer'
    assert coo_agent._persona['occupation']['organization'] == 'Global Supply Chain Corp'
    
    # Test decision making parameters
    decision_making = coo_agent._persona['decision_making']
    assert 'strategic_vision' in decision_making
    assert 'risk_tolerance' in decision_making
    assert 'adaptability' in decision_making

def test_strategic_decision_making(coo_agent, sample_world_state):
    """Test COO's strategic decision making."""
    decisions = coo_agent._make_strategic_decisions(sample_world_state)
    
    assert 'risk_mitigation' in decisions
    assert 'cost_optimization' in decisions
    assert 'resilience_focus' in decisions
    
    # Test decision values are within valid ranges
    assert 0 <= decisions['risk_mitigation'] <= 1
    assert 0 <= decisions['cost_optimization'] <= 1
    assert 0 <= decisions['resilience_focus'] <= 1

def test_regional_analysis(coo_agent, sample_world_state):
    """Test COO's regional performance analysis."""
    analysis = coo_agent._analyze_regional_performance(sample_world_state)
    
    assert 'NORTH_AMERICA' in analysis
    assert 'EUROPE' in analysis
    
    for region in [Region.NORTH_AMERICA, Region.EUROPE]:
        region_analysis = analysis[region.name]
        assert 'performance_score' in region_analysis
        assert 'risk_level' in region_analysis
        assert 'efficiency_score' in region_analysis
        assert 0 <= region_analysis['performance_score'] <= 1
        assert 0 <= region_analysis['risk_level'] <= 1
        assert 0 <= region_analysis['efficiency_score'] <= 1

def test_custom_act(coo_agent, sample_world_state):
    """Test COO's custom act behavior."""
    messages = coo_agent.custom_act(sample_world_state, return_actions=True)
    
    assert isinstance(messages, list)
    assert len(messages) > 0
    
    # Verify strategic decisions are reflected in messages
    assert any("strategic decision" in msg.lower() for msg in messages)
    assert any("risk" in msg.lower() for msg in messages)
    assert any("cost" in msg.lower() for msg in messages)

def test_resilience_strategy(coo_agent, sample_world_state):
    """Test COO's resilience strategy formulation."""
    strategy = coo_agent._formulate_resilience_strategy(sample_world_state)
    
    assert 'supplier_diversification' in strategy
    assert 'inventory_strategy' in strategy
    assert 'transportation_flexibility' in strategy
    assert 'regional_balance' in strategy
    
    # Test strategy components are within valid ranges
    assert 0 <= strategy['supplier_diversification'] <= 1
    assert 0 <= strategy['inventory_strategy'] <= 1
    assert 0 <= strategy['transportation_flexibility'] <= 1
    assert 0 <= strategy['regional_balance'] <= 1

def test_error_handling(coo_agent):
    """Test COO's error handling."""
    # Test with invalid world state
    message = coo_agent.custom_act(None)
    assert "Error in COO" in message
    
    # Test with missing required fields
    invalid_state = {'risk_exposure': 0.5}  # Missing most fields
    message = coo_agent.custom_act(invalid_state)
    assert "Error in COO" in message 