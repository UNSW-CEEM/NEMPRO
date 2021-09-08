import pandas as pd
from NEMPRO import historical_inputs, planner
import plotly.graph_objects as go
from plotly.subplots import make_subplots

raw_data_cache = 'E:/nem_data'

# Build data set for calibrating the dispatch planner's price forecasting model.
start_time_historical_data = '2020/01/01 00:00:00'
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

demand_data = demand_data.loc[:, ['SETTLEMENTDATE', 'nsw-demand']]
historical_data = pd.merge(price_data, demand_data, on='SETTLEMENTDATE')
historical_data = historical_data.reset_index(drop=True)
historical_data['interval'] = historical_data.index
#historical_data['hour'] = historical_data['SETTLEMENTDATE'].dt.hour
historical_data = historical_data.drop(columns=['SETTLEMENTDATE'])
historical_data['nsw-energy-fleet-dispatch'] = 0.0


# Build data set for running the dispatch planner's price forecasting model.
start_time_forward_data = '2020/01/07 00:00:00'
end_time_forward_data = '2020/01/14 00:00:00'

demand_data = historical_inputs.get_residual_demand(start_time_forward_data,
                                                    end_time_forward_data,
                                                    raw_data_cache)

demand_data = demand_data.loc[:, ['SETTLEMENTDATE', 'nsw-demand']]
demand_data = demand_data.reset_index(drop=True)
forward_data = demand_data.copy()
forward_data['interval'] = demand_data.index
#forward_data['hour'] = forward_data['SETTLEMENTDATE'].dt.hour
forward_data = forward_data.drop(columns=['SETTLEMENTDATE'])
forward_data['nsw-energy-fleet-dispatch'] = 0.0

f = planner.Forecaster()
f.train(historical_data, train_sample_fraction=0.005, target_col='nsw-energy')
forecast = f.price_forecast(forward_data, region='nsw', market='nsw-energy', min_delta=0, max_delta=6000,
                            steps=6)

price_data = historical_inputs.get_regional_prices(start_time_forward_data,
                                                   end_time_forward_data,
                                                   raw_data_cache)
price_data = price_data.reset_index(drop=True)
price_data['interval'] = price_data.index

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=forecast['interval'], y=forecast[0], name='forecast'))
fig.add_trace(go.Scatter(x=forecast['interval'], y=forecast[1000], name='forecast-1000'))
fig.add_trace(go.Scatter(x=price_data['interval'], y=price_data['nsw-energy'], name='hist'))
fig.add_trace(go.Scatter(x=demand_data.index, y=demand_data['nsw-demand'], name='hist'), secondary_y=True)
fig.write_html('forecast.html', auto_open=True)
