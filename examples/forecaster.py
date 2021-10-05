from NEMPRO import historical_inputs, planner
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pandas as pd


def run_forecast_and_plot_comparison_to_historical(
        start_time, training_peroid_length_weeks, forecast_length_weeks, raw_data_cache):

    # Build data set for calibrating the dispatch planner's price forecasting model.
    end_time_training = start_time
    end_time_training_obj = datetime.strptime(start_time, '%Y/%m/%d %H:%M:%S')
    start_time_training = end_time_training_obj - timedelta(weeks=training_peroid_length_weeks)
    start_time_training = datetime.isoformat(start_time_training).replace('-', '/').replace('T', ' ')

    training_data = historical_inputs.get_model_training_data(start_time=start_time_training,
                                                              end_time=end_time_training,
                                                              region='nsw',
                                                              raw_data_cache=raw_data_cache)

    # Exclude high price events to improve model performance. Note the column nsw-energy refers to the spot price
    # for energy in the nsw region.
    training_data = training_data[(training_data['nsw-energy'] < 300.0) &
                                  (training_data['nsw-energy'] > 0.0)]

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
    f.train(training_data, train_sample_fraction=0.1, target_col='nsw-energy')
    forecast = f.price_forecast_with_generation_sensitivities(forward_data, region='nsw', market='energy', min_delta=-1000,
                                                              max_delta=1000, steps=2)

    # Get actual historical price data for forecast period.
    price_data = historical_inputs.get_regional_prices(start_time_forward_data,
                                                       end_time_forward_data,
                                                       raw_data_cache)
    price_data = price_data.reset_index(drop=True)
    price_data['interval'] = price_data.index

    forecast = pd.merge(forecast, price_data.loc[:, ['interval', 'SETTLEMENTDATE']], on='interval')

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=price_data['SETTLEMENTDATE'], y=price_data['nsw-energy'], name='Historical prices'))
    fig.add_trace(go.Scatter(x=forecast['SETTLEMENTDATE'], y=forecast[0], name='Forecast prices'))
    fig.add_trace(go.Scatter(x=forecast['SETTLEMENTDATE'], y=forecast[-1000], name='Forecast minus 1 GW generation'))
    fig.add_trace(go.Scatter(x=forecast['SETTLEMENTDATE'], y=forecast[1000], name='Forecast plus 1 GW generation'))
    fig.update_xaxes(title="Time")
    fig.update_yaxes(title="Price ($/MWh)")
    fig.write_html('forecast.html', auto_open=True)


raw_data_cache = 'E:/nem_data'

run_forecast_and_plot_comparison_to_historical(start_time='2020/01/07 00:00:00', training_peroid_length_weeks=4,
                                               forecast_length_weeks=1, raw_data_cache=raw_data_cache)
run_forecast_and_plot_comparison_to_historical(start_time='2020/04/07 00:00:00', training_peroid_length_weeks=4,
                                               forecast_length_weeks=1, raw_data_cache=raw_data_cache)
run_forecast_and_plot_comparison_to_historical(start_time='2020/06/07 00:00:00', training_peroid_length_weeks=4,
                                               forecast_length_weeks=1, raw_data_cache=raw_data_cache)
run_forecast_and_plot_comparison_to_historical(start_time='2020/09/07 00:00:00', training_peroid_length_weeks=4,
                                               forecast_length_weeks=1, raw_data_cache=raw_data_cache)