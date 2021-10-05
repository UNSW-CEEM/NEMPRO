from NEMPRO import planner, units, historical_inputs
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

raw_data_cache = 'E:/nem_data'

# Build data set for calibrating the dispatch planner's price forecasting model.
start_time_historical_data = '2019/12/01 00:00:00'
end_time_historical_data = '2020/01/07 00:00:00'

# Get data to train forecasting model
training_data = historical_inputs.get_model_training_data(start_time=start_time_historical_data,
                                                          end_time=end_time_historical_data,
                                                          region='nsw',
                                                          raw_data_cache=raw_data_cache)

# Exclude high price events to improve model performance. Note the column nsw-energy refers to the spot price
# for energy in the nsw region.
training_data = training_data[(training_data['nsw-energy'] < 300.0) &
                              (training_data['nsw-energy'] > 0.0)]

# Need to include the historical dispatch of the fleet to be optimise. As this is a fictitious example we will just
# set the historical dispatch to zero
training_data['nsw-energy-fleet-dispatch'] = 0.0

# Build data set for running the dispatch planner's price forecasting model.
start_time_forward_data = '2020/01/07 00:00:00'
end_time_forward_data = '2020/01/14 00:00:00'

forward_data = historical_inputs.get_forward_data_for_forecast(start_time=start_time_forward_data,
                                                               end_time=end_time_forward_data,
                                                               raw_data_cache=raw_data_cache)

# Get price forecast traces with sensitivities for different fleet generation levels.
f = planner.Forecaster()
f.train(training_data, train_sample_fraction=0.1, target_col='nsw-energy')
price_forecast = f.price_forecast_with_generation_sensitivities(
    forward_data, region='nsw', market='energy', min_delta=-1000, max_delta=1000, steps=40)

# Get historical price data for forecast period.
price_data = historical_inputs.get_regional_prices(start_time_forward_data,
                                                   end_time_forward_data,
                                                   raw_data_cache)
price_data = price_data.reset_index(drop=True)
price_data['interval'] = price_data.index

# Create dispatch planner.
price_impact_planner = planner.DispatchPlanner(dispatch_interval=5, planning_horizon=len(forward_data.index))

# Add unit commitment model to dispatch planner, units are in MW
battery_storage = units.GenericUnit(price_impact_planner, initial_dispatch=0.0)
battery_storage.set_service_region('energy', 'nsw')
battery_storage.add_to_market_energy_flow(capacity=1000.0)
battery_storage.add_from_market_energy_flow(capacity=1000.0)
battery_storage.add_storage(mwh=4000.0, initial_mwh=2000.0, output_capacity=1000.0, input_capacity=1000.0,
                            output_efficiency=0.9, input_efficiency=0.9)

price_impact_planner.add_regional_market('nsw', 'energy', forecast=price_forecast)
price_impact_planner.optimise()

dispatch_plan = battery_storage.get_dispatch()

price_forecast = pd.merge(price_forecast, price_data.loc[:, ['interval', 'SETTLEMENTDATE']], on='interval')
dispatch_plan = pd.merge(dispatch_plan, price_data.loc[:, ['interval', 'SETTLEMENTDATE']], on='interval')

# Plot a comparison of dispatch planning with and without the price taker assumption.
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=price_forecast['SETTLEMENTDATE'], y=price_forecast[0],
                         name='Base case price forecast'))
fig.add_trace(go.Scatter(x=dispatch_plan['SETTLEMENTDATE'], y=dispatch_plan['net_dispatch'],
                         name='Dispatch plan'), secondary_y=True)
fig.update_xaxes(title="Time")
fig.update_yaxes(title="Price ($/MWh)", secondary_y=False)
fig.update_yaxes(title="Dispatch (MW)", secondary_y=True)
fig.write_html('images/battery_arbitrage_planning_1000MW.html', auto_open=True)
fig.write_image('images/battery_arbitrage_planning_1000MW.png')