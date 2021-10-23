from NEMPRO import historical_inputs, planner
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pandas as pd
from nemosis import dynamic_data_compiler


def run_forecast_and_plot_comparison_to_historical(
        start_time, training_peroid_length_weeks, forecast_length_weeks, raw_data_cache, units, region):

    price_column = '{}-energy'.format(region)

    # Build data set for calibrating the dispatch planner's price forecasting model.
    end_time_training = start_time
    end_time_training_obj = datetime.strptime(start_time, '%Y/%m/%d %H:%M:%S')
    start_time_training = end_time_training_obj - timedelta(weeks=training_peroid_length_weeks)
    start_time_training = datetime.isoformat(start_time_training).replace('-', '/').replace('T', ' ')

    training_data = historical_inputs.get_model_training_data(start_time=start_time_training,
                                                              end_time=end_time_training,
                                                              region=region,
                                                              raw_data_cache=raw_data_cache)

    # Exclude high price events to improve model performance. Note the column nsw-energy refers to the spot price
    # for energy in the nsw region.
    training_data = training_data[(training_data[price_column] < 1000.0) &
                                  (training_data[price_column] > 0.0)]

    # Build data set for running the dispatch planner's price forecasting model.
    start_time_forward_data = start_time
    start_time_forward_data_obj = datetime.strptime(start_time, '%Y/%m/%d %H:%M:%S')
    end_time_forward_data = start_time_forward_data_obj + timedelta(weeks=forecast_length_weeks)
    end_time_forward_data = datetime.isoformat(end_time_forward_data).replace('-', '/').replace('T', ' ')

    forward_data = historical_inputs.get_forward_data_for_forecast(start_time=start_time_forward_data,
                                                                   end_time=end_time_forward_data,
                                                                   raw_data_cache=raw_data_cache)

    # Train forecasting model and run forecast
    f = planner.Forecaster()
    f.train(training_data, train_sample_fraction=0.1, target_col=price_column)
    forecast = f.price_forecast_with_generation_sensitivities(forward_data, region=region, market='energy',
                                                              min_delta=-750, max_delta=0, steps=1)

    # Get actual historical price data for forecast period.
    price_data = historical_inputs.get_regional_prices(start_time_forward_data,
                                                       end_time_forward_data,
                                                       raw_data_cache)
    price_data = price_data.reset_index(drop=True)
    price_data['interval'] = price_data.index

    results = pd.merge(forecast, price_data.loc[:, ['interval', 'SETTLEMENTDATE', 'nsw-energy']], on='interval')
    results = pd.merge(results, forward_data, on='interval')

    plant_dispatch = dynamic_data_compiler(start_time_forward_data, end_time_forward_data, 'DISPATCH_UNIT_SCADA',
                                           raw_data_cache)
    plant_dispatch = plant_dispatch[plant_dispatch['DUID'].isin(units)]
    plant_dispatch = plant_dispatch.groupby('SETTLEMENTDATE', as_index=False).agg({'SCADAVALUE': 'sum'})

    price_data['moving_average'] = \
        price_data.loc[:, [price_column]].rolling(window=6, center=True).mean()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=price_data['SETTLEMENTDATE'], y=price_data[price_column], name='Historical prices'))
    fig.add_trace(go.Scatter(x=price_data['SETTLEMENTDATE'], y=price_data['moving_average'],
                             name='Historical moving average'))
    fig.add_trace(go.Scatter(x=results['SETTLEMENTDATE'], y=results[0], name='Forecast prices'))
    fig.add_trace(go.Scatter(x=results['SETTLEMENTDATE'], y=results[-750], name='Forecast minus 1 GW generation'))
    fig.add_trace(go.Scatter(x=plant_dispatch['SETTLEMENTDATE'], y=plant_dispatch['SCADAVALUE'],
                             name='Tripping Unit'), secondary_y=True)
    fig.update_xaxes(title="Time")
    fig.update_yaxes(title="Price ($/MWh)", secondary_y=False)
    fig.update_yaxes(title="Dispatch (MW)", secondary_y=True)
    fig.write_html('forecast.html', auto_open=True)

    results['bin'] = pd.qcut(results['{}-demand'.format(region)], 20, labels=False)


    trip_time = datetime.strptime('2018/04/18 20:48:00', '%Y/%m/%d %H:%M:%S')
    results_pre_trip = results[results['SETTLEMENTDATE'] <= trip_time]
    results_post_trip = results[results['SETTLEMENTDATE'] > trip_time]

    results_pre_trip = results_pre_trip.groupby('bin', as_index=False).mean()
    results_post_trip = results_post_trip.groupby('bin', as_index=False).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=results_pre_trip[0], x=results_pre_trip['bin'], name='Baseline forecast'))
    fig.add_trace(go.Scatter(y=results_pre_trip[-750], x=results_pre_trip['bin'], name='Forecast with tripping plant'))
    fig.add_trace(go.Scatter(y=results_pre_trip['nsw-energy'], x=results_pre_trip['bin'], name='Historical prices'))
    fig.write_html('demand_price_relationship_pre_trip.html', auto_open=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=results_post_trip[0], x=results_post_trip['bin'], name='Baseline forecast'))
    fig.add_trace(go.Scatter(y=results_post_trip[-750], x=results_post_trip['bin'], name='Forecast with tripping plant'))
    fig.add_trace(go.Scatter(y=results_pre_trip['nsw-energy'], x=results_pre_trip['bin'],
                         name='Historical prices pre trip'))
    fig.add_trace(go.Scatter(y=results_post_trip['nsw-energy'], x=results_post_trip['bin'],
                         name='Historical prices post trip'))
    fig.write_html('demand_price_relationship_post_trip.html', auto_open=True)


raw_data_cache = 'E:/nem_data'

# The Kogan Creek 750 MW generator tripped at 8:28 pm on th 18/04/2018. No other subsequent trips.
# https://wattclarity.com.au/articles/2018/04/when-the-nems-largest-generating-unit-trips/

run_forecast_and_plot_comparison_to_historical(start_time='2018/04/16 00:00:00', training_peroid_length_weeks=4,
                                               forecast_length_weeks=1, raw_data_cache=raw_data_cache,
                                               units=['KPP_1'], region='qld')
