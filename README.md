# Supply Chain Resilience Optimization Simulation

This project implements a Monte Carlo simulation to evaluate supply chain resilience improvements in a global manufacturing context. The simulation models various supply chain strategies and their impact on resilience, cost, and performance metrics.

## Overview

The simulation models a global supply chain with the following components:

- Geographic regions (North America, Europe, East Asia, Southeast Asia, South Asia)
- Supply chain agents (COO, Regional Managers, Suppliers, Logistics Providers, Production Facilities)
- External events (Weather, Geopolitical, Market)
- Resilience strategies (Supplier Diversification, Dynamic Inventory, Flexible Transportation, Regional Flexibility)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd supply-chain-simulation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the simulation:
```bash
python supply_chain_simulation.py
```

The simulation will:
1. Run a baseline scenario
2. Run scenarios with individual resilience strategies
3. Run a scenario with all strategies combined
4. Generate visualizations and export results

## Outputs

The simulation generates:
1. `supply_chain_simulation_results.png`: Visual comparison of metrics across scenarios
2. `supply_chain_simulation_results.csv`: Detailed numerical results

## Key Metrics

The simulation tracks the following metrics, each normalized to a scale of 0.0 to 1.0:

### Core Metrics

1. **Resilience Score**
   - A composite metric measuring overall supply chain resilience
   - Calculation: Weighted average of:
     * 30% Service Level
     * 30% (1 - Risk Exposure)
     * 20% Flexibility Score
     * 20% Quality Score

2. **Recovery Time**
   - Measures how quickly the supply chain can recover from disruptions
   - Calculation: Average of Lead Time and (1 - Flexibility Score)
   - Lower values indicate faster recovery

3. **Service Level**
   - Measures ability to meet customer demand
   - Adjusted based on regional reports and operational performance
   - Impacted by inventory levels, delivery performance, and supplier reliability

### Cost Metrics

4. **Total Cost**
   - Aggregate measure of all supply chain costs
   - Influenced by strategic decisions and operational efficiency
   - Updated based on COO's strategic assessment and regional performance

5. **Inventory Cost**
   - Cost of maintaining inventory across regions
   - Adjusted based on:
     * Regional inventory levels
     * Storage requirements
     * Dynamic inventory management effectiveness

6. **Transportation Cost**
   - Cost of moving goods across the network
   - Influenced by:
     * Mode selection (air, ocean, ground)
     * Route optimization
     * Fuel costs and capacity utilization

### Risk Metrics

7. **Risk Exposure**
   - Overall supply chain vulnerability to disruptions
   - Calculated from:
     * Supplier diversity
     * Geographic distribution
     * External event impacts

8. **Supplier Risk**
   - Measures reliability and stability of supplier base
   - Affected by:
     * Supplier diversification
     * Regional stability
     * Supplier performance history

9. **Transportation Risk**
   - Risk of delays or disruptions in logistics
   - Based on:
     * Mode reliability
     * Route complexity
     * Regional infrastructure quality

### Performance Metrics

10. **Lead Time**
    - Time from order placement to delivery
    - Impacted by:
      * Transportation efficiency
      * Supplier performance
      * Regional infrastructure

11. **Flexibility Score**
    - Ability to adapt to changes and disruptions
    - Influenced by:
      * Production capacity
      * Supplier alternatives
      * Transportation mode options

12. **Quality Score**
    - Measure of product and service quality
    - Based on:
      * Supplier quality performance
      * Production facility standards
      * Transportation handling

### Metric Updates

Metrics are continuously updated based on:
- COO's strategic assessments
- Regional manager reports
- Supplier performance updates
- External events
- Agent interactions and decisions

All metrics are bounded between 0.0 (worst) and 1.0 (best) through normalization after each update.

## Customization

You can customize the simulation by modifying:
1. `DEFAULT_CONFIG` in `supply_chain.py`: Adjust simulation parameters
2. Agent behaviors in `supply_chain.py`: Modify decision-making logic
3. Metrics calculation in `supply_chain_simulation.py`: Change how metrics are computed

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
