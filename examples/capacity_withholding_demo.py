import pandas as pd
from NEMPRO import planner, units, historical_inputs
import plotly.graph_objects as go
from plotly.subplots import make_subplots

raw_data_cache = 'E:/nem_data'

# Build data set for calibrating the dispatch planner's price forecasting model.
start_time_historical_data = '2019/12/01 00:00:00'
end_time_historical_data = '2020/01/07 00:00:00'

tech_availability = historical_inputs.get_tech_operating_capacities(start_time_historical_data,
                                                                    end_time_historical_data,
                                                                    raw_data_cache)

price_data = historical_inputs.get_regional_prices(start_time_historical_data,
                                                   end_time_historical_data,
                                                   raw_data_cache)

price_data = price_data.loc[:, ['SETTLEMENTDATE', 'nsw-energy']]

demand_data = historical_inputs.get_residual_demand(start_time_historical_data,
                                                    end_time_historical_data,
                                                    raw_data_cache)

historical_data = pd.merge(price_data, demand_data, on='SETTLEMENTDATE')
historical_data = historical_data.reset_index(drop=True)
historical_data['interval'] = historical_data.index
historical_data['hour'] = historical_data['SETTLEMENTDATE'].dt.hour
historical_data = historical_data.drop(columns=['SETTLEMENTDATE'])
historical_data['nsw-energy-fleet-dispatch'] = 0.0


# Build data set for running the dispatch planner's price forecasting model.
start_time_forward_data = '2020/01/07 00:00:00'
end_time_forward_data = '2020/01/14 00:00:00'

demand_data = historical_inputs.get_residual_demand(start_time_forward_data,
                                                    end_time_forward_data,
                                                    raw_data_cache)

demand_data = demand_data.reset_index(drop=True)
forward_data = demand_data.copy()
forward_data['interval'] = demand_data.index
forward_data['hour'] = forward_data['SETTLEMENTDATE'].dt.hour
forward_data = forward_data.drop(columns=['SETTLEMENTDATE'])

historical_data = historical_data[(historical_data['nsw-energy'] < 300.0) &
                                  (historical_data['nsw-energy'] > 0.0)]

p = planner.DispatchPlanner(dispatch_interval=5, historical_data=historical_data, forward_data=forward_data,
                            demand_delta_steps=50)

u = units.GenericUnit(p, initial_dispatch=0.0)
u.set_service_region('energy', 'nsw')
u.add_to_market_energy_flow(capacity=10000.0)
u.add_primary_energy_source(capacity=10000.0, cost=50.0)

p.add_regional_market('nsw', 'energy')

p.optimise()

dispatch = u.get_dispatch()

forecast = p.nominal_price_forecast['nsw-energy']

price_data = historical_inputs.get_regional_prices(start_time_forward_data,
                                                   end_time_forward_data,
                                                   raw_data_cache)
price_data = price_data.reset_index(drop=True)
price_data['interval'] = price_data.index

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=forecast['interval'], y=forecast[0], name='forecast'))
fig.add_trace(go.Scatter(x=forecast['interval'], y=forecast[1000], name='forecast-1000'))
fig.add_trace(go.Scatter(x=forecast['interval'], y=forecast[10000], name='forecast-6000'))
fig.add_trace(go.Scatter(x=price_data['interval'], y=price_data['nsw-energy'], name='hist'))
fig.add_trace(go.Scatter(x=dispatch.index, y=dispatch['net_dispatch'], name='hist'), secondary_y=True)
fig.write_html('capacity_witholding_demo.html', auto_open=True)