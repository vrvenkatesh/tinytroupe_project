from typing import Dict, Any, Union, List
from datetime import datetime, timedelta

from tinytroupe.agent import TinyPerson
from models.enums import OrderStatus, Region

class COOAgent(TinyPerson):
    """Chief Operations Officer agent responsible for overseeing supply chain operations."""
    
    def __init__(self, name: str, config: Dict[str, Any], simulation_id: str):
        """Initialize the COO agent with required parameters."""
        super().__init__(name=name)
        self.config = config
        self.simulation_id = simulation_id  # Store for _post_init
        
        # Initialize persona
        self._persona = {
            'occupation': {
                'title': 'Chief Operating Officer',
                'organization': 'Global Supply Chain Corp',
                'responsibilities': [
                    'Global supply chain oversight',
                    'Strategic decision making',
                    'Risk management',
                    'Performance optimization'
                ]
            },
            'decision_making': {
                'strategic_vision': 0.9,
                'risk_tolerance': 0.7,
                'adaptability': 0.8,
                'analytical_skills': 0.9
            },
            'personality': {
                'leadership': 0.9,
                'communication': 0.8,
                'problem_solving': 0.9,
                'stress_management': 0.8
            }
        }

    def _post_init(self, **kwargs):
        """Post-initialization setup."""
        super()._post_init(**kwargs)
        if hasattr(self, 'simulation_id'):
            self._simulation_id = self.simulation_id
            delattr(self, 'simulation_id')
    
    def _make_strategic_decisions(self, world_state: Dict[str, Any]) -> Dict[str, float]:
        """Make strategic decisions based on current world state."""
        if not world_state:
            return {
                'risk_mitigation': 0.5,
                'cost_optimization': 0.5,
                'resilience_focus': 0.5
            }
        
        try:
            risk_exposure = world_state.get('risk_exposure', 0.5)
            cost_pressure = world_state.get('cost_pressure', 0.5)
            demand_volatility = world_state.get('demand_volatility', 0.6)
            
            return {
                'risk_mitigation': min(1.0, risk_exposure * 1.2),
                'cost_optimization': min(1.0, cost_pressure * 1.1),
                'resilience_focus': min(1.0, demand_volatility * 1.3)
            }
        except Exception as e:
            return {
                'risk_mitigation': 0.5,
                'cost_optimization': 0.5,
                'resilience_focus': 0.5
            }
    
    def _analyze_regional_performance(self, world_state: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Analyze performance across different regions."""
        analysis = {}
        
        if not world_state or 'regional_metrics' not in world_state:
            return {
                region.name: {
                    'performance_score': 0.5,
                    'risk_level': 0.5,
                    'efficiency_score': 0.5
                } for region in Region
            }
        
        regional_metrics = world_state.get('regional_metrics', {})
        
        for region in Region:
            # Try both the enum name and value, and uppercase name
            metrics = None
            if region.name in regional_metrics:
                metrics = regional_metrics[region.name]
            elif region.value in regional_metrics:
                metrics = regional_metrics[region.value]
            elif region.name.upper() in regional_metrics:
                metrics = regional_metrics[region.name.upper()]
            
            if metrics:
                analysis[region.name.upper()] = {
                    'performance_score': (
                        metrics['efficiency'] * 0.4 +
                        metrics['quality'] * 0.3 +
                        metrics['flexibility'] * 0.3
                    ),
                    'risk_level': (
                        metrics['risk'] * 0.3 +
                        metrics['supply_risk'] * 0.4 +
                        metrics['congestion'] * 0.3
                    ),
                    'efficiency_score': metrics['efficiency']
                }
            else:
                analysis[region.name.upper()] = {
                    'performance_score': 0.5,
                    'risk_level': 0.5,
                    'efficiency_score': 0.5
                }
        
        return analysis
    
    def _formulate_resilience_strategy(self, world_state: Dict[str, Any]) -> Dict[str, float]:
        """Formulate resilience strategy based on current conditions."""
        if not world_state:
            return {
                'supplier_diversification': 0.5,
                'inventory_strategy': 0.5,
                'transportation_flexibility': 0.5,
                'regional_balance': 0.5
            }
        
        try:
            risk_exposure = world_state.get('risk_exposure', 0.5)
            supply_risk = world_state.get('supply_risk', 0.5)
            flexibility_req = world_state.get('flexibility_requirement', 0.7)
            
            return {
                'supplier_diversification': min(1.0, supply_risk * 1.2),
                'inventory_strategy': min(1.0, risk_exposure * 1.1),
                'transportation_flexibility': min(1.0, flexibility_req * 1.2),
                'regional_balance': min(1.0, (risk_exposure + supply_risk) / 2 * 1.1)
            }
        except Exception as e:
            return {
                'supplier_diversification': 0.5,
                'inventory_strategy': 0.5,
                'transportation_flexibility': 0.5,
                'regional_balance': 0.5
            }
    
    def custom_act(self, world_state: Dict[str, Any] = None, return_actions: bool = False) -> Union[str, List[str]]:
        """Enhanced act method for COO with proper interaction recording."""
        if not world_state:
            error_msg = "Error in COO analysis: No world state provided"
            return [error_msg] if return_actions else error_msg
        
        try:
            # Validate required fields
            required_fields = ['risk_exposure', 'cost_pressure', 'demand_volatility', 'supply_risk']
            missing_fields = [field for field in required_fields if field not in world_state]
            if missing_fields:
                error_msg = f"Error in COO analysis: Missing required fields: {', '.join(missing_fields)}"
                return [error_msg] if return_actions else error_msg
            
            # Make strategic decisions
            decisions = self._make_strategic_decisions(world_state)
            
            # Analyze regional performance
            regional_analysis = self._analyze_regional_performance(world_state)
            
            # Formulate resilience strategy
            resilience_strategy = self._formulate_resilience_strategy(world_state)
            
            messages = [
                f"Strategic decisions made: Risk mitigation at {decisions['risk_mitigation']:.2f}, "
                f"Cost optimization at {decisions['cost_optimization']:.2f}",
                f"Regional analysis completed for {len(regional_analysis)} regions",
                f"Resilience strategy formulated with focus on supplier diversification "
                f"({resilience_strategy['supplier_diversification']:.2f}) and regional balance "
                f"({resilience_strategy['regional_balance']:.2f})"
            ]
            
            return messages if return_actions else "\n".join(messages)
            
        except Exception as e:
            error_msg = f"Error in COO analysis: {str(e)}"
            return [error_msg] if return_actions else error_msg
    
    def handle_critical_escalation(self, escalation: Dict[str, Any], world_state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle critical issues escalated from regional managers."""
        try:
            response = {
                'mitigation_strategy': {},
                'risk_assessment': {},
                'corrective_action': {}
            }
            
            if 'severity' in escalation:
                response['risk_assessment'] = {
                    'severity_level': escalation['severity'],
                    'impact_scope': 'global' if escalation['severity'] > 0.8 else 'regional',
                    'urgency': 'high' if escalation['severity'] > 0.7 else 'medium'
                }
            
            if 'type' in escalation:
                if escalation['type'] == 'invalid_order':
                    response['corrective_action'] = {
                        'action': 'order_validation_review',
                        'priority': 'high',
                        'assigned_to': 'regional_manager'
                    }
                else:
                    response['mitigation_strategy'] = {
                        'type': 'risk_mitigation',
                        'actions': ['increase_monitoring', 'adjust_thresholds'],
                        'timeline': 'immediate'
                    }
            
            return response
            
        except Exception as e:
            return {
                'error': str(e),
                'status': 'failed',
                'timestamp': datetime.now().isoformat()
            }
    
    def evaluate_strategic_proposal(self, proposal: Dict[str, Any], world_state: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate strategic proposals from regional managers."""
        try:
            if not proposal.get('proposal') or not proposal.get('justification'):
                return {
                    'approved': False,
                    'modifications': {},
                    'reason': 'Incomplete proposal'
                }
            
            # Evaluate based on current world state and proposal details
            risk_level = proposal.get('risk_level', 0.5)
            current_risk = world_state.get('risk_exposure', 0.5)
            
            approved = risk_level < 0.8 and current_risk < 0.7
            
            return {
                'approved': approved,
                'modifications': {
                    'risk_mitigation_required': risk_level > 0.6,
                    'phased_implementation': risk_level > 0.5,
                    'additional_monitoring': True
                },
                'evaluation_metrics': {
                    'risk_assessment': risk_level,
                    'strategic_alignment': 0.8,
                    'resource_efficiency': 0.7
                }
            }
            
        except Exception as e:
            return {
                'approved': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            } 