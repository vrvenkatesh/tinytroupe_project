"""Functions for creating and managing the simulation world."""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import random
import uuid

from tinytroupe.environment.tiny_world import TinyWorld as World
from models.enums import Region, OrderStatus, TransportationMode
from models.order import Order
from agents import COOAgent, RegionalManagerAgent, SupplierAgent, ExternalEventAgent

def create_simulation_world(config: Dict[str, Any]) -> World:
    """Create a new simulation world with all necessary agents."""
    world = World()
    
    # Set random seed for reproducibility
    random.seed(config['simulation']['seed'])
    
    # Create COO
    coo = COOAgent("COO", config['coo'], str(uuid.uuid4()))
    world.add_agent(coo)
    
    # Create regional managers for each region
    regional_managers = []
    for region in Region:
        manager = RegionalManagerAgent(
            name=f"Manager_{region.value}",
            config=config['regional_manager'],
            simulation_id=str(uuid.uuid4()),
            region=region
        )
        world.add_agent(manager)
        regional_managers.append(manager)
    
    # Create suppliers for each region
    suppliers = []
    for region in Region:
        for i in range(config['simulation']['suppliers_per_region']):
            supplier = SupplierAgent(
                name=f"Supplier_{region.value}_{i}",
                config=config['supplier'],
                simulation_id=str(uuid.uuid4()),
                region=region,
                supplier_type='tier1'
            )
            world.add_agent(supplier)
            suppliers.append(supplier)
    
    # Create external event generators
    for event_type in ['weather', 'geopolitical', 'market']:
        event_generator = ExternalEventAgent(
            name=f"{event_type.capitalize()}EventGenerator",
            config=config['external_events'][event_type],
            simulation_id=str(uuid.uuid4()),
            event_type=event_type
        )
        world.add_agent(event_generator)
    
    return world

def simulate_supply_chain_operation(world: World, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single time step of the supply chain simulation."""
    current_datetime = datetime.now()
    
    # Generate new orders
    new_orders = _generate_orders(current_datetime, config)
    active_orders = world.state.get('active_orders', []) + new_orders
    world.state['active_orders'] = active_orders
    
    # Process orders through regional managers
    regional_managers = [agent for agent in world.agents if isinstance(agent, RegionalManagerAgent)]
    for manager in regional_managers:
        manager.manage_region(world.state)
    
    # Update order statuses
    completed_orders = []
    remaining_orders = []
    
    for order in active_orders:
        if order.status == OrderStatus.DELIVERED:
            completed_orders.append(order)
        elif _check_delivery(order, current_datetime, config):
            order.update_status(OrderStatus.DELIVERED, current_datetime)
            completed_orders.append(order)
        else:
            if _check_delays(order, current_datetime, config):
                order.update_status(OrderStatus.DELAYED, current_datetime)
            remaining_orders.append(order)
    
    # Update world state
    world.state['active_orders'] = remaining_orders
    world.state['completed_orders'] = world.state.get('completed_orders', []) + completed_orders
    world.state['current_datetime'] = current_datetime
    
    # Calculate metrics
    metrics = _calculate_order_based_metrics(
        completed_orders=world.state['completed_orders'],
        active_orders=remaining_orders,
        config=config
    )
    
    return metrics

def _generate_orders(current_datetime: datetime, config: Dict[str, Any]) -> List[Order]:
    """Generate new orders based on base demand and random factors."""
    orders = []
    base_demand = config['simulation']['base_demand']
    
    for source in Region:
        for dest in Region:
            if source != dest:
                # Random demand variation
                demand = int(base_demand * (0.5 + random.random()))
                
                for _ in range(demand):
                    delivery_time = _estimate_delivery_time(source, dest, config)
                    expected_delivery = current_datetime + timedelta(days=delivery_time)
                    
                    order = Order(
                        id=str(uuid.uuid4()),
                        product_type="Standard",
                        quantity=random.randint(1, 100),
                        source_region=source,
                        destination_region=dest,
                        creation_time=current_datetime,
                        expected_delivery_time=expected_delivery
                    )
                    orders.append(order)
    
    return orders

def _check_delivery(order: Order, current_datetime: datetime, config: Dict[str, Any]) -> bool:
    """Check if an order should be marked as delivered."""
    if order.status == OrderStatus.IN_TRANSIT:
        transit_time = (current_datetime - order.creation_time).days
        expected_time = _estimate_delivery_time(order.source_region, order.destination_region, config)
        return transit_time >= expected_time
    return False

def _check_delays(order: Order, current_datetime: datetime, config: Dict[str, Any]) -> bool:
    """Check if an order should be marked as delayed."""
    if order.status in [OrderStatus.PRODUCTION, OrderStatus.IN_TRANSIT]:
        current_time = (current_datetime - order.creation_time).days
        expected_time = _estimate_delivery_time(order.source_region, order.destination_region, config)
        return current_time > expected_time * 1.5
    return False

def _estimate_delivery_time(source: Region, dest: Region, config: Dict[str, Any]) -> int:
    """Estimate delivery time between two regions."""
    base_time = config['production_facility']['base_production_time']
    distance_factor = 2 if source.value in ['East Asia', 'Southeast Asia'] and dest.value in ['North America', 'Europe'] else 1
    return base_time * distance_factor

def _calculate_order_based_metrics(completed_orders: List[Order], active_orders: List[Order], config: Dict[str, Any]) -> Dict[str, float]:
    """Calculate various metrics based on order status and performance."""
    total_orders = len(completed_orders) + len(active_orders)
    if total_orders == 0:
        return {
            'completion_rate': 0.0,
            'on_time_delivery_rate': 0.0,
            'average_delay': 0.0
        }
    
    completion_rate = len(completed_orders) / total_orders
    on_time_deliveries = sum(1 for order in completed_orders if order.is_on_time())
    on_time_rate = on_time_deliveries / len(completed_orders) if completed_orders else 0.0
    
    total_delay = sum(order.delay_time for order in completed_orders + active_orders)
    average_delay = total_delay / total_orders
    
    return {
        'completion_rate': completion_rate,
        'on_time_delivery_rate': on_time_rate,
        'average_delay': average_delay
    } 