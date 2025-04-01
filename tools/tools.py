from tinytroupe.tools.tiny_tool import TinyTool
from typing import Dict, Any
from disruption_manager import DisruptionManager

# Initialize the disruption manager
disruption_manager = DisruptionManager()


class SupplierSelectionEvaluator(TinyTool):
    def __init__(self):
        super().__init__("supplier_selection_evaluator")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates supplier diversification strategies, accounting for disruptions.
        """
        inputs = disruption_manager.apply_disruptions(inputs)
        risk_adjusted_score = inputs["reliability"] - inputs["geopolitical_risk"] + (1 / inputs["cost"])
        return {
            "recommended_supplier": "High" if risk_adjusted_score > 0.7 else "Medium" if risk_adjusted_score > 0.4 else "Low"}


class InventoryOptimizer(TinyTool):
    def __init__(self):
        super().__init__("inventory_optimizer")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimizes inventory levels based on risk factors and disruptions.
        """
        inputs = disruption_manager.apply_disruptions(inputs)
        buffer = inputs["component_criticality"] * inputs["disruption_probability"]
        return {"recommended_stock_level": inputs["demand_forecast"] + buffer}


class TransportationRouteOptimizer(TinyTool):
    def __init__(self):
        super().__init__("transportation_route_optimizer")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determines optimal transportation routes based on real-time conditions and disruptions.
        """
        inputs = disruption_manager.apply_disruptions(inputs)
        if inputs["congestion_level"] < 0.3:
            return {"recommended_route": "Direct"}
        elif inputs["cost"] < 500:
            return {"recommended_route": "Alternative Route"}
        else:
            return {"recommended_route": "Expedited Shipping"}


class RegionalRiskAnalyzer(TinyTool):
    def __init__(self):
        super().__init__("regional_risk_analyzer")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assesses regional risks, factoring in disruptions.
        """
        inputs = disruption_manager.apply_disruptions(inputs)
        risk_score = (10 - inputs["political_stability"]) + (5 - inputs["labor_conditions"]) + inputs["disaster_risk"]
        return {"regional_risk_score": risk_score}


class ProductionTransferSimulator(TinyTool):
    def __init__(self):
        super().__init__("production_transfer_simulator")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tests feasibility of shifting production across facilities, adjusting for disruptions.
        """
        inputs = disruption_manager.apply_disruptions(inputs)
        feasibility = inputs["capacity"] * inputs["labor_availability"] / (1 + inputs["regulatory_constraints"])
        return {"transfer_feasibility": "High" if feasibility > 0.7 else "Medium" if feasibility > 0.4 else "Low"}


class SupplyChainResilienceScorecard(TinyTool):
    def __init__(self):
        super().__init__("supply_chain_resilience_scorecard")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates resilience score, incorporating disruption impacts.
        """
        inputs = disruption_manager.apply_disruptions(inputs)
        score = (inputs["service_level"] * 10) - (inputs["cost_impact"] * 5) - (inputs["recovery_time"] * 2)
        return {"resilience_score": max(0, min(score, 100))}
