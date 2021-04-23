import pandas as pd
import random
from datetime import datetime, timedelta
import itertools
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from NEMPRO import historical_inputs, planner


def get_test_start_intervals(number):
    start_time = datetime(year=2017, month=1, day=1, hour=0, minute=0)
    end_time = datetime(year=2020, month=12, day=31, hour=0, minute=0)
    difference = end_time - start_time
    difference_in_5_min_intervals = difference.days * 12 * 24
    random.seed(1)
    intervals = random.sample(range(1, difference_in_5_min_intervals), number)
    times = [start_time + timedelta(minutes=5 * i) for i in intervals]
    times_formatted = [datetime_to_aemo_format(t) for t in times]
    return times_formatted


def datetime_to_aemo_format(date_time):
    return date_time.isoformat().replace('T', ' ').replace('-', '/')


end_training_periods = get_test_start_intervals(10)

t0 = time.time()
c = 0

for end_training_period in end_training_periods:

    c += 1

    try:

        raw_data_cache = 'C:/Users/nick/Documents/nem_data'

        # Build data set for calibrating the dispatch planner's price forecasting model.
        end_time = datetime.strptime(end_training_period, '%Y/%m/%d %H:%M:%S')
        start_time_historical_data = datetime_to_aemo_format(end_time - timedelta(days=7))
        end_time_historical_data = end_training_period

        start_time_forward_data = end_training_period
        end_time_forward_data = datetime_to_aemo_format(end_time + timedelta(days=1))

        price_data = historical_inputs.get_regional_prices(start_time_historical_data,
                                                           end_time_historical_data,
                                                           raw_data_cache)

        price_data = price_data.loc[:, ['SETTLEMENTDATE', 'nsw-energy']]

        demand_data = historical_inputs.get_residual_demand(start_time_historical_data,
                                                            end_time_historical_data,
                                                            raw_data_cache)

        historical_data = pd.merge(price_data, demand_data, on='SETTLEMENTDATE')
        historical_data = historical_data.sort_values('SETTLEMENTDATE', ascending=False)
        historical_data = historical_data.reset_index(drop=True)
        historical_data['interval'] = historical_data.index
        historical_data = historical_data.sort_values('SETTLEMENTDATE')
        historical_data = historical_data.drop(columns=['SETTLEMENTDATE'])

        demand_data = historical_inputs.get_residual_demand(start_time_forward_data,
                                                            end_time_forward_data,
                                                            raw_data_cache)

        forward_data = demand_data.reset_index(drop=True)
        forward_data['interval'] = forward_data.index
        forward_data = forward_data.drop(columns=['SETTLEMENTDATE'])

        hist_price_data = historical_data.loc[:, ['interval', 'nsw-energy']]
        hist_regression_features = historical_data.loc[:,
                                   ['interval', 'qld-demand', 'nsw-demand', 'vic-demand', 'sa-demand',
                                    'tas-demand']]

        f = planner.MultiMarketForecaster()
        f.train(hist_price_data, hist_regression_features, ['nsw-demand', 'qld-demand'])

        fleet_dispatch_delta = {'nsw-energy': [-1000, 0, 1000]}

        forecast = f.multi_region_price_forecast(forward_data, fleet_dispatch_delta)

        forecast = forecast.pivot(index='interval', columns='nsw-energy-fleet-delta', values='nsw-energy')

        price_data = historical_inputs.get_regional_prices(start_time_forward_data,
                                                           end_time_forward_data,
                                                           raw_data_cache)
        price_data = price_data.reset_index(drop=True)
        price_data['interval'] = price_data.index


        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast[0], name='baseline forecast'))
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast[-1000], name='extra giga watt demand'))
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast[1000], name='less giga watt demand'))
        fig.add_trace(go.Scatter(x=price_data['interval'], y=price_data['nsw-energy'], name='hist'))
        fig.show()
    except:
        pass




