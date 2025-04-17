from typing import Dict, Any, List
from tinytroupe.tools.tiny_tool import TinyTool, ToolRegistry

class SupplyChainTools:
    """Factory class for creating supply chain specific tools."""
    
    @staticmethod
    def create_coo_tools() -> List[TinyTool]:
        """Create tools for COO agents."""
        return [
            TinyTool(
                name="review_global_performance",
                description="Review global supply chain performance metrics",
                function=lambda world_state, **kwargs: {
                    "service_level": world_state.get("service_level", 0),
                    "resilience_score": world_state.get("resilience_score", 0),
                    "total_orders": len(world_state.get("orders", [])),
                }
            ),
            TinyTool(
                name="optimize_network",
                description="Optimize the supply chain network configuration",
                function=lambda world_state, **kwargs: {
                    "optimization_result": "Network optimization completed",
                    "affected_regions": world_state.get("regions", []),
                }
            )
        ]
    
    @staticmethod
    def create_regional_manager_tools() -> List[TinyTool]:
        """Create tools for Regional Manager agents."""
        return [
            TinyTool(
                name="monitor_regional_inventory",
                description="Monitor inventory levels in the region",
                function=lambda world_state, region, **kwargs: {
                    "region": region,
                    "inventory_levels": world_state.get("inventory", {}).get(region, {}),
                    "stock_outs": world_state.get("stock_outs", {}).get(region, 0),
                }
            ),
            TinyTool(
                name="manage_regional_orders",
                description="Manage and prioritize orders in the region",
                function=lambda world_state, region, **kwargs: {
                    "region": region,
                    "active_orders": [
                        order for order in world_state.get("orders", [])
                        if order.get("region") == region
                    ],
                }
            )
        ]
    
    @staticmethod
    def create_supplier_tools() -> List[TinyTool]:
        """Create tools for Supplier agents."""
        return [
            TinyTool(
                name="check_capacity",
                description="Check current production capacity",
                function=lambda world_state, supplier_id, **kwargs: {
                    "supplier_id": supplier_id,
                    "capacity": world_state.get("capacity", {}).get(supplier_id, 0),
                    "utilization": world_state.get("utilization", {}).get(supplier_id, 0),
                }
            ),
            TinyTool(
                name="process_order",
                description="Process a new production order",
                function=lambda world_state, supplier_id, order, **kwargs: {
                    "supplier_id": supplier_id,
                    "order_id": order.get("id"),
                    "status": "processing",
                    "estimated_completion": world_state.get("current_time", 0) + 
                        world_state.get("base_production_time", 1),
                }
            )
        ]

    @staticmethod
    def create_logistics_tools() -> List[TinyTool]:
        """Create tools for Logistics agents."""
        return [
            TinyTool(
                name="calculate_route",
                description="Calculate optimal route for delivery",
                function=lambda world_state, origin, destination, **kwargs: {
                    "origin": origin,
                    "destination": destination,
                    "distance": world_state.get("distances", {}).get((origin, destination), 0),
                    "estimated_time": world_state.get("transit_times", {}).get((origin, destination), 0),
                }
            ),
            TinyTool(
                name="track_shipment",
                description="Track status of a shipment",
                function=lambda world_state, shipment_id, **kwargs: {
                    "shipment_id": shipment_id,
                    "status": world_state.get("shipments", {}).get(shipment_id, {}).get("status"),
                    "current_location": world_state.get("shipments", {}).get(shipment_id, {}).get("location"),
                }
            )
        ]
    
    @classmethod
    def create_registry(cls, agent_type: str) -> ToolRegistry:
        """Create a tool registry for a specific agent type."""
        tools_map = {
            "COO": cls.create_coo_tools,
            "RegionalManager": cls.create_regional_manager_tools,
            "Supplier": cls.create_supplier_tools,
            "Logistics": cls.create_logistics_tools,
        }
        
        if agent_type not in tools_map:
            raise ValueError(f"Unknown agent type: {agent_type}")
            
        tools = tools_map[agent_type]()
        return ToolRegistry(tools) 