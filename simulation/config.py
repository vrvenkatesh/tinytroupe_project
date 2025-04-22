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
        'cost_sensitivity': 0.6,
        'order_batch_size': 20,  # Added: Number of orders to process in each batch
        'order_processing_interval': 24,  # Added: Hours between order processing
        'regional_demand_weights': {  # Added: Weights for demand distribution
            'North America': 0.3,
            'Europe': 0.3,
            'East Asia': 0.2,
            'Southeast Asia': 0.1,
            'South Asia': 0.1
        },
        'regional_production_costs': {  # Added: Production costs per region
            'North America': 100,
            'Europe': 120,
            'East Asia': 80,
            'Southeast Asia': 90,
            'South Asia': 85
        }
    },
    'supplier': {
        'reliability': 0.8,
        'quality': 0.9,
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
        'capacity': {  # Production capacity settings per region
            'North America': 200,  # Higher capacity for larger market
            'Europe': 150,         # Medium capacity
            'East Asia': 250,      # Highest capacity for manufacturing hub
            'Southeast Asia': 200,  # High capacity for growing market
            'South Asia': 150      # Medium capacity for emerging market
        }
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