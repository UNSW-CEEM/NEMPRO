from NEMPRO import planner, units, historical_inputs
import plotly.graph_objects as go
from plotly.subplots import make_subplots

raw_data_cache = 'E:/nem_data'

# Build data set for calibrating the dispatch planner's price forecasting model.
start_time_historical_data = '2019/12/01 00:00:00'
end_time_historical_data = '2020/01/07 00:00:00'

training_data = historical_inputs.get_model_training_data(start_time=start_time_historical_data,
                                                          end_time=end_time_historical_data,
                                                          region='nsw',
                                                          raw_data_cache=raw_data_cache)

#
training_data = training_data[(training_data['nsw-energy'] < 300.0) &
                              (training_data['nsw-energy'] > 0.0)]

# Need to include the historical dispatch of the fleet to be optimise. As this is a fictous example we will just
# set the historical dispatch to zero
training_data['nsw-energy-fleet-dispatch'] = 0.0


# Build data set for running the dispatch planner's price forecasting model.
start_time_forward_data = '2020/01/07 00:00:00'
end_time_forward_data = '2020/01/14 00:00:00'

forward_data = historical_inputs.get_forward_data_for_forecast(start_time=start_time_forward_data,
                                                               end_time=end_time_forward_data,
                                                               raw_data_cache=raw_data_cache)


# Get historical price data for forecast period.
price_data = historical_inputs.get_regional_prices(start_time_forward_data,
                                                   end_time_forward_data,
                                                   raw_data_cache)
price_data = price_data.reset_index(drop=True)
price_data['interval'] = price_data.index


# Do dispatch planning assuming price taker assumption. Planner uses actual historical price data and assumes
# fleet dispatch does not impact price.
price_taker_planner = planner.DispatchPlanner(dispatch_interval=5, forward_data=price_data)

# Add unit commitment model to dispatch planner
battery_storage = units.GenericUnit(price_taker_planner, initial_dispatch=0.0)
battery_storage.set_service_region('energy', 'nsw')
battery_storage.add_to_market_energy_flow(capacity=1000.0)
battery_storage.add_from_market_energy_flow(capacity=1000.0)
battery_storage.add_storage(mwh=4000.0, initial_mwh=2000.0, output_capacity=1000.0, input_capacity=1000.0,
                            output_efficiency=0.9, input_efficiency=0.9)

price_taker_planner.add_regional_market('nsw', 'energy')
price_taker_planner.optimise()

price_taker_dispatch = battery_storage.get_dispatch()

# Do dispatch planning without assuming price taker assumption. Planner uses a forecasting model to estimate the impact
# dispatch will have on price.
price_impact_planner = planner.DispatchPlanner(dispatch_interval=5, historical_data=training_data,
                                               forward_data=forward_data, demand_delta_steps=30)

# Add unit commitment model to dispatch planner
battery_storage = units.GenericUnit(price_impact_planner, initial_dispatch=0.0)
battery_storage.set_service_region('energy', 'nsw')
battery_storage.add_to_market_energy_flow(capacity=1000.0)
battery_storage.add_from_market_energy_flow(capacity=1000.0)
battery_storage.add_storage(mwh=4000.0, initial_mwh=2000.0, output_capacity=1000.0, input_capacity=1000.0,
                            output_efficiency=0.9, input_efficiency=0.9)

price_impact_planner.add_regional_market('nsw', 'energy')
price_impact_planner.optimise()

price_impact_dispatch = battery_storage.get_dispatch()

forecast = price_impact_planner.nominal_price_forecast['nsw-energy']


# Plot a comparison of dispatch planning with and without the price taker assumption.
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=forecast['interval'], y=forecast[-10], name='Base case price impact planner forecast'))
fig.add_trace(go.Scatter(x=price_data['interval'], y=price_data['nsw-energy'], name='Historical price'))
fig.add_trace(go.Scatter(x=price_taker_dispatch.index, y=price_taker_dispatch['net_dispatch'],
                         name='Dispatch with price taker planning'), secondary_y=True)
fig.add_trace(go.Scatter(x=price_impact_dispatch.index, y=price_impact_dispatch['net_dispatch'],
                         name='Dispatch with price impact planning'), secondary_y=True)
fig.write_html('battery_arbitrage_planning.html', auto_open=True)