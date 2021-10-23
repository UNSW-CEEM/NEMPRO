from NEMPRO import planner, units, historical_inputs
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

raw_data_cache = 'E:/nem_data'

# Build data set for calibrating the dispatch planner's price forecasting model.
start_time_historical_data = '2020/01/01 00:00:00'
end_time_historical_data = '2020/02/07 00:00:00'

# Get data to train forecasting model
training_data = historical_inputs.get_model_training_data(start_time=start_time_historical_data,
                                                          end_time=end_time_historical_data,
                                                          region='qld',
                                                          raw_data_cache=raw_data_cache)

# Exclude high price events to improve model performance. Note the column nsw-energy refers to the spot price
# for energy in the nsw region.
training_data = training_data[(training_data['qld-energy'] < 300.0) &
                              (training_data['qld-energy'] > 0.0)]

# Build data set for running the dispatch planner's price forecasting model.
start_time_forward_data = '2020/02/07 00:00:00'
end_time_forward_data = '2020/02/14 00:00:00'

forward_data = historical_inputs.get_forward_data_for_forecast(start_time=start_time_forward_data,
                                                               end_time=end_time_forward_data,
                                                               raw_data_cache=raw_data_cache)

# We also need to forecast the BAU fleet dispatch
historical_fleet_dispatch = historical_inputs.get_fleet_dispatch(start_time=start_time_historical_data,
                                                     end_time=end_time_historical_data,
                                                     fleet_units=['CALL_B_1', 'CALL_B_2',
                                                                  # 'GSTONE1', 'GSTONE2',
                                                                  # 'GSTONE3', 'GSTONE4', 'GSTONE5', 'GSTONE6',
                                                                  'KPP_1'],
                                                     region='qld',
                                                     raw_data_cache=raw_data_cache)

fleet_dispatch_training_data = pd.merge(training_data, historical_fleet_dispatch, on='interval')
fleet_dispatch_training_data = fleet_dispatch_training_data.drop(columns=['qld-energy'])

f = planner.Forecaster()
f.train(fleet_dispatch_training_data, train_sample_fraction=0.5, target_col='qld-energy-fleet-dispatch')
fleet_dispatch_forecast = f.single_trace_forecast(forward_data)
forward_data = pd.merge(forward_data, fleet_dispatch_forecast, on='interval')

# Get price forecast traces with sensitivities for different fleet generation levels.
f = planner.Forecaster()
f.train(training_data, train_sample_fraction=0.5, target_col='qld-energy')
price_forecast = f.price_forecast_with_generation_sensitivities(
    forward_data, region='qld', market='energy', min_delta=-3500, max_delta=3500, steps=50)

# Get historical price data for forecast period.
price_data = historical_inputs.get_regional_prices(start_time_forward_data,
                                                   end_time_forward_data,
                                                   raw_data_cache)
price_data = price_data.reset_index(drop=True)
price_data['interval'] = price_data.index

# Create dispatch planner.
cs_energy_planner = planner.DispatchPlanner(dispatch_interval=5, planning_horizon=len(forward_data.index))

# Add unit commitment model to dispatch planner, units are in MW
callide_b = units.GenericUnit(cs_energy_planner, initial_dispatch=140.0 * 2)
callide_b.set_service_region('energy', 'qld')
callide_b.add_to_market_energy_flow(capacity=350.0 * 2)
callide_b.add_primary_energy_source(capacity=350.0 * 2, cost=24)
callide_b.add_unit_hard_minimum_operating_level(min_loading=140 * 2)
# callide_b.add_ramp_rates_to_energy_flow_to_market(ramp_up_rate=3.17 * 2, ramp_down_rate=3.17 * 2)

# gladstone = units.GenericUnit(cs_energy_planner, initial_dispatch=110.0 * 6)
# gladstone.set_service_region('energy', 'qld')
# gladstone.add_to_market_energy_flow(capacity=280.0 * 6)
# gladstone.add_primary_energy_source(capacity=280.0 * 6, cost=28)
# gladstone.add_unit_hard_minimum_operating_level(min_loading=110 * 6)
# gladstone.add_ramp_rates_to_energy_flow_to_market(ramp_up_rate=4.83 * 6, ramp_down_rate=5.0 * 6)

kogan = units.GenericUnit(cs_energy_planner, initial_dispatch=744.0)
kogan.set_service_region('energy', 'qld')
kogan.add_to_market_energy_flow(capacity=744.0)
kogan.add_primary_energy_source(capacity=744.0, cost=16)
kogan.add_unit_hard_minimum_operating_level(min_loading=300)
# kogan.add_ramp_rates_to_energy_flow_to_market(ramp_up_rate=3.5, ramp_down_rate=3.5)

cs_energy_planner.add_regional_market('qld', 'energy', forecast=price_forecast)
cs_energy_planner.optimise()

callide_b_dispatch = callide_b.get_dispatch()
# gladstone_dispatch = gladstone.get_dispatch()
kogan_dispatch = kogan.get_dispatch()

dipatch_plan = pd.concat([callide_b_dispatch, kogan_dispatch])
dipatch_plan = dipatch_plan.groupby('interval', as_index=False).sum()
forward_data = pd.merge(forward_data, dipatch_plan, on='interval')
forward_data['qld-demand'] = forward_data['qld-demand'] - \
                             (forward_data['net_dispatch'] - forward_data['qld-energy-fleet-dispatch'])
forward_data = forward_data.drop(columns=['net_dispatch', 'qld-energy-fleet-dispatch'])
modified_forecast_price = f.single_trace_forecast(forward_data)

price_forecast = pd.merge(price_forecast, price_data.loc[:, ['interval', 'SETTLEMENTDATE']], on='interval')
callide_b_dispatch = pd.merge(callide_b_dispatch, price_data.loc[:, ['interval', 'SETTLEMENTDATE']], on='interval')
#gladstone_dispatch = pd.merge(gladstone_dispatch, price_data.loc[:, ['interval', 'SETTLEMENTDATE']], on='interval')
kogan_dispatch = pd.merge(kogan_dispatch, price_data.loc[:, ['interval', 'SETTLEMENTDATE']], on='interval')
fleet_dispatch_forecast = pd.merge(fleet_dispatch_forecast, price_data.loc[:, ['interval', 'SETTLEMENTDATE']],
                                   on='interval')
historical_fleet_dispatch = pd.merge(historical_fleet_dispatch, price_data.loc[:, ['interval', 'SETTLEMENTDATE']],
                                     on='interval')
modified_forecast_price = pd.merge(modified_forecast_price, price_data.loc[:, ['interval', 'SETTLEMENTDATE']],
                                     on='interval')

# Plot a comparison of dispatch planning with and without the price taker assumption.
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=price_forecast['SETTLEMENTDATE'], y=price_forecast[0],
                         name='Base case price forecast'))
fig.add_trace(go.Scatter(x=modified_forecast_price['SETTLEMENTDATE'], y=modified_forecast_price['qld-energy'],
                         name='Price forecast with dispatch plan'))
fig.add_trace(go.Scatter(x=callide_b_dispatch['SETTLEMENTDATE'], y=callide_b_dispatch['net_dispatch'],
                         name='Callide dispatch plan', stackgroup='one'), secondary_y=True)
# fig.add_trace(go.Scatter(x=gladstone_dispatch['SETTLEMENTDATE'], y=gladstone_dispatch['net_dispatch'],
#                          name='Gladstone dispatch plan', stackgroup='one'), secondary_y=True)
fig.add_trace(go.Scatter(x=kogan_dispatch['SETTLEMENTDATE'], y=kogan_dispatch['net_dispatch'],
                         name='Kogan Creek dispatch', stackgroup='one'), secondary_y=True)
fig.add_trace(go.Scatter(x=fleet_dispatch_forecast['SETTLEMENTDATE'],
                         y=fleet_dispatch_forecast['qld-energy-fleet-dispatch'],
                         name='Forecast Dispatch'), secondary_y=True)
fig.add_trace(go.Scatter(x=historical_fleet_dispatch['SETTLEMENTDATE'],
                         y=historical_fleet_dispatch['qld-energy-fleet-dispatch'],
                         name='Historical Dispatch'), secondary_y=True)

fig.update_xaxes(title="Time")
fig.update_yaxes(title="Price ($/MWh)", secondary_y=False)
fig.update_yaxes(title="Dispatch (MW)", secondary_y=True)
fig.write_html('images/cs_energy_demo.html', auto_open=True)
fig.write_image('images/cs_energy_demo.png')
