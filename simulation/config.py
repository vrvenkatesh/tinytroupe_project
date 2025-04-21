"""Default configuration for the supply chain simulation."""

DEFAULT_CONFIG = {
    'simulation': {
        'seed': 42,
        'monte_carlo_iterations': 100,
        'suppliers_per_region': 3,
        'time_steps': 365,  # One year of daily operations
        'base_demand': 10,  # Base demand per region pair
    },
    'coo': {
        'risk_aversion': 0.7,
        'cost_sensitivity': 0.5,
        'strategic_vision': 0.8,
        'initial_metrics': {
            'resilience_score': 0.6,
            'recovery_time': 0.7,
            'service_level': 0.8,
            'total_cost': 0.7,
            'inventory_cost': 0.6,
            'transportation_cost': 0.7,
            'risk_exposure': 0.5,
            'supplier_risk': 0.4,
            'transportation_risk': 0.5,
            'lead_time': 0.7,
            'flexibility_score': 0.6,
            'quality_score': 0.8
        }
    },
    'regional_manager': {
        'local_expertise': 0.8,
        'adaptability': 0.7,
        'communication_skills': 0.6,
        'cost_sensitivity': 0.6
    },
    'supplier': {
        'reliability': 0.8,
        'quality_score': 0.9,
        'cost_efficiency': 0.7,
        'diversification_enabled': False,
    },
    'logistics': {
        'reliability': 0.8,
        'cost_efficiency': 0.7,
        'flexibility': 0.6,
        'flexible_routing_enabled': False,
    },
    'production_facility': {
        'efficiency': 0.8,
        'quality_control': 0.9,
        'flexibility': 0.7,
        'regional_flexibility_enabled': False,
        'base_production_time': 3,  # Base time in days for production
    },
    'inventory_management': {
        'base_stock_level': 100,
        'safety_stock_factor': 1.5,
        'dynamic_enabled': False,
    },
    'external_events': {
        'weather': {
            'frequency': 0.1,
            'severity': 0.5,
        },
        'geopolitical': {
            'frequency': 0.05,
            'severity': 0.7,
        },
        'market': {
            'frequency': 0.2,
            'severity': 0.4,
        },
    },
} 