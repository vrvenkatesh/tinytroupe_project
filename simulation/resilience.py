import copy
from enum import Enum
from typing import Dict, Any
from .state import SimulationState

class ResponseType(Enum):
    INCREASE_CAPACITY = "increase_capacity"
    OPTIMIZE_WORKFLOW = "optimize_workflow"
    REDUCE_COMPLEXITY = "reduce_complexity"
    ENHANCE_COORDINATION = "enhance_coordination"

def apply_adaptive_response(state: SimulationState, response_type: ResponseType) -> SimulationState:
    """
    Apply an adaptive response to the current simulation state.
    
    Args:
        state: Current simulation state
        response_type: Type of response to apply
        
    Returns:
        Updated simulation state
    """
    # Create a copy of the state to avoid modifying the original
    new_state = SimulationState(**state.__dict__)
    
    if response_type == ResponseType.INCREASE_CAPACITY:
        new_state.team_capacity = min(1.0, state.team_capacity * 1.2)
        new_state.resource_availability = min(1.0, state.resource_availability * 1.15)
        
    elif response_type == ResponseType.OPTIMIZE_WORKFLOW:
        new_state.workflow_efficiency = min(1.0, state.workflow_efficiency * 1.15)
        new_state.error_rate = max(0.01, state.error_rate * 0.85)
        
    elif response_type == ResponseType.REDUCE_COMPLEXITY:
        new_state.task_complexity = max(0.1, state.task_complexity * 0.8)
        new_state.error_rate = max(0.01, state.error_rate * 0.9)
        
    elif response_type == ResponseType.ENHANCE_COORDINATION:
        new_state.communication_quality = min(1.0, state.communication_quality * 1.2)
        new_state.team_coordination = min(1.0, state.team_coordination * 1.15)
    
    # Update derived metrics after applying changes
    new_state.update_metrics()
    return new_state 