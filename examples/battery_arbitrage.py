import pandas as pd
from NEMPRO import historical_inputs, planner
import plotly.graph_objects as go
from plotly.subplots import make_subplots

raw_data_cache = 'nem_data'

# Build data set for calibrating the dispatch planner's price forecasting model.
start_time_historical_data = '2020/11/01 00:00:00'
end_time_historical_data = '2021/01/01 00:00:00'

price_data = historical_inputs.get_regional_prices(start_time_historical_data,
                                                   end_time_historical_data,
                                                   raw_data_cache)

demand_data = historical_inputs.get_regional_demand(start_time_historical_data,
                                                    end_time_historical_data,
                                                    raw_data_cache)

tech_availability = historical_inputs.get_tech_operating_capacities(start_time_historical_data,
                                                                    end_time_historical_data,
                                                                    raw_data_cache)

historical_data = pd.merge(price_data, demand_data, on='SETTLEMENTDATE')
historical_data = pd.merge(historical_data, tech_availability, on='SETTLEMENTDATE')
historical_data = historical_data.reset_index(drop=True)
historical_data['interval'] = historical_data.index
historical_data = historical_data.drop(columns=['SETTLEMENTDATE'])
historical_data['nsw-energy-fleet-dispatch'] = 0.0


# Build data set for running the dispatch planner's price forecasting model.
start_time_forward_data = '2021/01/01 00:00:00'
end_time_forward_data = '2021/02/01 00:00:00'

demand_data = historical_inputs.get_regional_demand(start_time_forward_data,
                                                    end_time_forward_data,
                                                    raw_data_cache)

tech_availability = historical_inputs.get_tech_operating_capacities(start_time_forward_data,
                                                                    end_time_forward_data,
                                                                    raw_data_cache)

forward_data = pd.merge(demand_data, tech_availability, on='SETTLEMENTDATE')
forward_data = forward_data.reset_index(drop=True)
forward_data['interval'] = forward_data.index
forward_data = forward_data.drop(columns=['SETTLEMENTDATE'])

p = planner.DispatchPlanner(dispatch_interval=5, historical_data=historical_data,
                            forward_data=forward_data, demand_delta_steps=50)

p.add_unit('big_battery', 'nsw', initial_mw=0.0)
p.add_unit_to_market_flow('big_battery', capacity=1000)
p.add_market_to_unit_flow('big_battery', capacity=1000)
p.add_storage('big_battery', input_capacity=1000, output_capacity=1000, mwh=1000,
              initial_mwh=500, input_efficiency=0.9, output_efficiency=0.9)

p.add_regional_market('nsw', 'energy')

p.optimise()

dispatch = p.get_unit_energy_flows('big_battery')

forecast = p.get_nominal_price_forecast('nsw', 'energy')

dispatch.to_csv('big_battery_dispatch.csv')

price_data = historical_inputs.get_regional_prices(start_time_forward_data,
                                                   end_time_forward_data,
                                                   raw_data_cache)
price_data = price_data.reset_index(drop=True)
price_data['interval'] = price_data.index

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=dispatch['interval'], y=dispatch['net_dispatch'], name='net dispatch'), secondary_y=False)
fig.add_trace(go.Scatter(x=price_data['interval'], y=price_data['nsw-energy'], name='price'), secondary_y=True)
fig.add_trace(go.Scatter(x=forecast['interval'], y=forecast[0], name='price forecast'), secondary_y=True)
fig.show()
