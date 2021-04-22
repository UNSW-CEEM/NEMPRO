import matplotlib.pyplot as plt

import pandas as pd
from datetime import datetime, timedelta

from NEMPRO import historical_inputs, planner


def datetime_to_aemo_format(date_time):
    return date_time.isoformat().replace('T', ' ').replace('-', '/')

raw_data_cache = 'C:/Users/nick/Documents/nem_data'

end_training_period = '2020/10/19 16:35:00'
training_length = 120

end_time = datetime.strptime(end_training_period, '%Y/%m/%d %H:%M:%S')
start_time_historical_data = datetime_to_aemo_format(end_time - timedelta(days=training_length))
end_time_historical_data = end_training_period

start_time_forward_data = end_training_period
end_time_forward_data = datetime_to_aemo_format(end_time + timedelta(days=1))

price_data = historical_inputs.get_regional_prices(start_time_historical_data,
                                                   end_time_historical_data,
                                                   raw_data_cache)

price_data = price_data.loc[:, ['SETTLEMENTDATE', 'nsw-energy', 'qld-energy']]

demand_data = historical_inputs.get_residual_demand(start_time_historical_data,
                                                    end_time_historical_data,
                                                    raw_data_cache)

historical_data = pd.merge(price_data, demand_data, on='SETTLEMENTDATE')
historical_data.sort_values('SETTLEMENTDATE')
historical_data = historical_data.reset_index(drop=True)
historical_data['interval'] = historical_data.index
historical_data.sort_values('SETTLEMENTDATE')
historical_data = historical_data.drop(columns=['SETTLEMENTDATE'])

demand_data = historical_inputs.get_residual_demand(start_time_forward_data,
                                                    end_time_forward_data,
                                                    raw_data_cache)

forward_data = demand_data.reset_index(drop=True)
forward_data['interval'] = forward_data.index
forward_data = forward_data.drop(columns=['SETTLEMENTDATE'])

hist_price_data = historical_data.loc[:, ['interval', 'nsw-energy', 'qld-energy']]
hist_regression_features = historical_data.loc[:, ['interval', 'qld-demand', 'nsw-demand', 'vic-demand', 'sa-demand',
                                                   'tas-demand']]

f = planner.MultiMarketForecaster()
f.train(hist_price_data, hist_regression_features, ['nsw-demand', 'qld-demand'])

fleet_dispatch_delta = {'nsw-energy': [0, 500, 1000, 1500, 2000, 2500, 3000],
                        'qld-energy': [0, 500, 1000, 1500, 2000, 2500, 3000]}

forecast = f.multi_region_price_forecast(forward_data, fleet_dispatch_delta)

first_interval_results = forecast[forecast['interval'] == 0]

nsw_dispatch = first_interval_results['nsw-energy-fleet-delta']
qld_dispatch = first_interval_results['qld-energy-fleet-delta']

revenue = nsw_dispatch * first_interval_results['nsw-energy'] + qld_dispatch * first_interval_results['qld-energy'] \
    + nsw_dispatch * -30.0 + qld_dispatch * -30.0

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter3D(nsw_dispatch, qld_dispatch, revenue, color="green")
plt.show()