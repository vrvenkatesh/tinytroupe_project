from dataclasses import dataclass
from typing import Dict, Any, List
import copy

@dataclass
class SimulationState:
    """Class representing the state of the supply chain simulation."""
    
    # Core metrics
    team_capacity: float = 1.0
    resource_availability: float = 1.0
    workflow_efficiency: float = 1.0
    task_completion_rate: float = 1.0
    task_complexity: float = 1.0
    error_rate: float = 0.1
    communication_quality: float = 1.0
    team_coordination: float = 1.0
    
    # Derived metrics
    completion_rate: float = 0.0
    on_time_delivery_rate: float = 0.0
    resilience_score: float = 0.0
    risk_level: float = 0.0
    average_delay: float = 0.0
    
    def update_metrics(self) -> None:
        """Update derived metrics based on core metrics."""
        # Calculate completion rate based on team capacity and workflow efficiency
        self.completion_rate = min(1.0, 
            self.team_capacity * self.workflow_efficiency * (1.0 - self.error_rate))
        
        # Calculate on-time delivery based on task completion and coordination
        self.on_time_delivery_rate = min(1.0,
            self.task_completion_rate * self.team_coordination)
        
        # Calculate resilience score based on resource availability and communication
        self.resilience_score = min(1.0,
            (self.resource_availability + self.communication_quality) / 2)
        
        # Calculate risk level based on task complexity and error rate
        self.risk_level = min(1.0,
            (self.task_complexity * self.error_rate))
        
        # Calculate average delay based on efficiency and coordination
        self.average_delay = max(0.0,
            3.0 * (2.0 - self.workflow_efficiency - self.team_coordination)) 