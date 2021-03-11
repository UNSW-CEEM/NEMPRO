import pandas as pd
from NEMPRO import historical_inputs, planner
import plotly.graph_objects as go
from plotly.subplots import make_subplots

raw_data_cache = 'C:/Users/nick/Documents/nem_data'

# Build data set for calibrating the dispatch planner's price forecasting model.
start_time_historical_data = '2020/01/01 00:00:00'
end_time_historical_data = '2021/01/01 00:00:00'

price_data = historical_inputs.get_regional_prices(start_time_historical_data,
                                                   end_time_historical_data,
                                                   raw_data_cache)
price_data = price_data.loc[:, ['SETTLEMENTDATE', 'nsw-energy']]

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
start_time_forward_data = '2021/01/15 18:00:00'
end_time_forward_data = '2021/01/15 18:05:00'

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
forward_data['nsw-energy-fleet-dispatch'] = 0.0

demand_delta = pd.DataFrame({
    'interval': [0, 0, 0, 0, 0, 0, 0],
    'delta': [0, 1000, 2000, 3000, 4000, 5000, 6000]
})

forward_data = pd.merge(forward_data, demand_delta, on='interval')

forward_data = forward_data.reset_index(drop=True)
forward_data['interval'] = forward_data.index

forward_data['nsw-demand'] = forward_data['nsw-demand'] - forward_data['delta']

f = planner.Forecaster()
f.train(historical_data, train_sample_fraction=0.10, target_col='nsw-energy')
forecast = f.base_forecast(forward_data.drop(columns='delta'))

forward_data = pd.merge(forward_data, forecast, on='interval')
forward_data = forward_data.loc[:, ['delta', 'nsw-energy']]
forward_data['revenue'] = forward_data['delta'] * forward_data['nsw-energy']
forward_data.to_csv('test_negative_revenue_threshold.csv')
print(forward_data)


#
# price_data = historical_inputs.get_regional_prices(start_time_forward_data,
#                                                    end_time_forward_data,
#                                                    raw_data_cache)
# price_data = price_data.reset_index(drop=True)
# price_data['interval'] = price_data.index
#
# fig = make_subplots(specs=[[{"secondary_y": True}]])
# fig.add_trace(go.Scatter(x=forecast['interval'], y=forecast[0], name='forecast'))
# fig.add_trace(go.Scatter(x=price_data['interval'], y=price_data['nsw-energy'], name='hist'))
# fig.show()
