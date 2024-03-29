# NEMPRO: National Electricity Market Portfolio Revenue Optimiser

## Introduction
NEMPRO is a behavoural model of particpants in the Australian National Electricity Market. The behavoural model assumes 
participants attempt to dispatch their generation portfolios in order maximise net revenue, revenue after operating 
expense. A key part of this optimisation is modelling the impact of the portfolio's dispatch on spot market prices. 
Additionally, constraints on portfolio dispatch such as minimum stable opperating levels, start up costs, minium up and 
down times, ramp rates, and storage charging and discharging are also modelled. However, each market region is modelled 
separately, i.e. the impact of portflio dispatch in one region on prices in another region is not modelled, and only 
dispatch in the energy spot market is modelled and optimised.

## Development stage
The project is currently at the technology demonstration stage of development, and does not include detailed 
documentation and user support features that you would expect in production software.

## Architecture
The functionality of NEMPRO is/will be devided into three core components: 1) a forecasting module that generates the 
baseline prices forecasts and senstivity forecasts that estimate the impact of additional generation on prices, 2) a 
planning module that uses the forecasts to optimse the dispatch plan of a given portfolio of generation assests, and 3) a 
bidding module that transaltes the dispatch plan into a set of bids. While working proto types of the forecasting and 
planning modules have been developed, work on the bidding module hasn't started.

## Implementation
The forecasting module has been implemented using the causalnex package (https://github.com/quantumblacklabs/causalnex). 
The planning module is implmented using linear programming, the mip python package is used for model formulation 
(https://github.com/coin-or/python-mip), and either the open-source CBC solver or commercial Gurobi solver can be used. 
Unit commitment constraints have been implemented based of Knueven et al's (2020) paper On Mixed Integer Programming 
Formulations for Unit Commitment.

## Examples

### Battery arbitrage
A simple example to demonstrate NEMPRO is the case of planning a dispatch schedule for a large battery. In the first 
chart below the dispatch plan for a 1000 MW battery with 4 h of storage is shown. The code for example is here:
https://github.com/UNSW-CEEM/NEMPRO/blob/master/examples/battery_arbitrage_planning.py. The plan optimiser discharges
at time of high price, and recharges at times of low price. However to minimise the impact of dispatch on price, and
maximise net revenue the optimiser does not always charge and discharge at full power, but spreads out charging and 
discharging more evenly. The second chart shows the same example but for 3000 MW battery with 4 h of storage.

|![Figure1](/examples/images/battery_arbitrage_planning_1000MW.png)|
|:--:|
|Dispatch plan for a 1000 MW battery with 4 h of storage|

|![Figure2](/examples/images/battery_arbitrage_planning_3000MW.png)|
|:--:|
|Dispatch plan for a 3000 MW battery with 4 h of storage|
