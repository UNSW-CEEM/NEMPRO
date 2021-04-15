import pandas as pd
from NEMPRO import historical_inputs, planner
import plotly.graph_objects as go
from plotly.subplots import make_subplots

raw_data_cache = 'C:/Users/nick/Documents/nem_data'

# Build data set for calibrating the dispatch planner's price forecasting model.
start_time_historical_data = '2020/01/09 00:00:00'
end_time_historical_data = '2020/02/16 00:00:00'

#tech_availability = historical_inputs.get_tech_operating_capacities(start_time_historical_data,
#                                                                    end_time_historical_data,
#                                                                    raw_data_cache)

price_data = historical_inputs.get_regional_prices(start_time_historical_data,
                                                   end_time_historical_data,
                                                   raw_data_cache)

price_data = price_data.loc[:, ['SETTLEMENTDATE', 'nsw-energy']]

demand_data = historical_inputs.get_residual_demand(start_time_historical_data,
                                                    end_time_historical_data,
                                                    raw_data_cache)


historical_data = pd.merge(price_data, demand_data, on='SETTLEMENTDATE')
#historical_data = pd.merge(historical_data, tech_availability, on='SETTLEMENTDATE')
historical_data.sort_values('SETTLEMENTDATE')
historical_data = historical_data.reset_index(drop=True)
historical_data['interval'] = historical_data.index
historical_data['hour'] = historical_data['SETTLEMENTDATE'].dt.hour
#historical_data['dayofyear'] = historical_data['SETTLEMENTDATE'].dt.dayofyear
#historical_data['dayofweek'] = historical_data['SETTLEMENTDATE'].dt.dayofweek
historical_data.sort_values('SETTLEMENTDATE', ascending=False)
historical_data = historical_data.drop(columns=['SETTLEMENTDATE'])
historical_data['nsw-energy-fleet-dispatch'] = 0.0



# Build data set for running the dispatch planner's price forecasting model.
start_time_forward_data = '2020/02/16 00:00:00'
end_time_forward_data = '2020/02/17 00:00:00'

demand_data = historical_inputs.get_residual_demand(start_time_forward_data,
                                                    end_time_forward_data,
                                                    raw_data_cache)

#tech_availability = historical_inputs.get_tech_operating_capacities(start_time_forward_data,
#                                                                    end_time_forward_data,
#                                                                    raw_data_cache)

#forward_data = pd.merge(demand_data, tech_availability, on='SETTLEMENTDATE')
forward_data = demand_data.reset_index(drop=True)
forward_data['interval'] = forward_data.index
forward_data['hour'] = forward_data['SETTLEMENTDATE'].dt.hour
#forward_data['dayofyear'] = forward_data['SETTLEMENTDATE'].dt.dayofyear
#forward_data['dayofweek'] = forward_data['SETTLEMENTDATE'].dt.dayofweek
forward_data = forward_data.drop(columns=['SETTLEMENTDATE'])
forward_data['nsw-energy-fleet-dispatch'] = 0.0

f = planner.Forecaster()
f.train(historical_data, train_sample_fraction=1, target_col='nsw-energy')
forecast = f.price_forecast(forward_data, region='nsw', market='nsw-energy', min_delta=0, max_delta=6000,
                            steps=10)

price_data = historical_inputs.get_regional_prices(start_time_forward_data,
                                                   end_time_forward_data,
                                                   raw_data_cache)
price_data = price_data.reset_index(drop=True)
price_data['interval'] = price_data.index

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=forecast['interval'], y=forecast[0], name='forecast'))
fig.add_trace(go.Scatter(x=price_data['interval'], y=price_data['nsw-energy'], name='hist'))
fig.show()

forward_data = pd.merge(price_data, forecast, on='interval')
forward_data['error'] = (forward_data[0] - forward_data['nsw-energy']).abs()
print(forward_data['error'].mean())
