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


end_training_periods = get_test_start_intervals(30)
training_lengths = [7]
alphas = [0.001]
betas = [0.5]
layers = [[5, 5]]
include_day_of_week = [False]
bins = [50]
train_set_size = [5000]

record_end_training_periods = []
record_training_lengths = []
record_alphas = []
record_betas = []
record_layers = []
record_include_day_of_week = []
record_bins = []
record_train_set_size = []

average_absolute_error = []
average_error = []
plus_one_giggawatts_price_change = []
plus_five_giggawatts_price_change = []
minus_one_giggawatts_price_change = []

t0 = time.time()
c = 0

for end_training_period, training_length, alpha, beta, layer, day_of_week, bin, set_size in \
        itertools.product(end_training_periods, training_lengths, alphas, betas, layers, include_day_of_week, bins,
                          train_set_size):
    c += 1

    try:
        raw_data_cache = 'C:/Users/nick/Documents/nem_data'

        # Build data set for calibrating the dispatch planner's price forecasting model.
        end_time = datetime.strptime(end_training_period, '%Y/%m/%d %H:%M:%S')
        start_time_historical_data = datetime_to_aemo_format(end_time - timedelta(days=training_length))
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
        historical_data.sort_values('SETTLEMENTDATE')
        historical_data = historical_data.reset_index(drop=True)
        historical_data['interval'] = historical_data.index
        historical_data['hour'] = historical_data['SETTLEMENTDATE'].dt.hour
        if day_of_week:
            historical_data['dayofweek'] = historical_data['SETTLEMENTDATE'].dt.day_of_week
        historical_data.sort_values('SETTLEMENTDATE', ascending=False)
        historical_data = historical_data.drop(columns=['SETTLEMENTDATE'])
        historical_data['nsw-energy-fleet-dispatch'] = 0.0

        demand_data = historical_inputs.get_residual_demand(start_time_forward_data,
                                                            end_time_forward_data,
                                                            raw_data_cache)

        forward_data = demand_data.reset_index(drop=True)
        forward_data['interval'] = forward_data.index
        forward_data['hour'] = forward_data['SETTLEMENTDATE'].dt.hour
        if day_of_week:
            forward_data['dayofweek'] = forward_data['SETTLEMENTDATE'].dt.day_of_week
        forward_data = forward_data.drop(columns=['SETTLEMENTDATE'])
        forward_data['nsw-energy-fleet-dispatch'] = 0.0

        f = planner.Forecaster()
        f.train(historical_data, train_sample_fraction=1, target_col='nsw-energy', alpha=alpha, beta=beta,
                layers=layer, bins=bin, sample_size=set_size)
        forecast = f.price_forecast(forward_data, region='nsw', market='nsw-energy',
                                    demand_delta=[-5000.0, -1000.0, 0.0, 1000.0])

        price_data = historical_inputs.get_regional_prices(start_time_forward_data,
                                                           end_time_forward_data,
                                                           raw_data_cache)
        price_data = price_data.reset_index(drop=True)
        price_data['interval'] = price_data.index

        error_data = pd.merge(price_data, forecast, on='interval')
        error_data['error'] = (error_data[0] - error_data['nsw-energy']).abs()

        record_end_training_periods.append(end_training_period)
        record_training_lengths.append(training_length)
        record_alphas.append(alpha)
        record_betas.append(beta)
        record_layers.append(layer)
        record_include_day_of_week.append(day_of_week)
        record_bins.append(bin)
        record_train_set_size.append(set_size)

        average_absolute_error.append((error_data[0] - error_data['nsw-energy']).abs().mean())
        average_error.append((error_data[0] - error_data['nsw-energy']).mean())
        plus_one_giggawatts_price_change.append((error_data[-1000] - error_data[0]).mean())
        plus_five_giggawatts_price_change.append((error_data[-5000] - error_data[0]).mean())
        minus_one_giggawatts_price_change.append((error_data[1000] - error_data[0]).mean())

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=error_data['interval'], y=error_data[0], name='forecast'))
        fig.add_trace(go.Scatter(x=error_data['interval'], y=error_data['nsw-energy'], name='hist'))
        fig.show()

    except:
        print(end_training_period)



