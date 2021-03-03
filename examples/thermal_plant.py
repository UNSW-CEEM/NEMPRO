import pandas as pd
from NEMPRO import historical_inputs, planner
import plotly.graph_objects as go
from plotly.subplots import make_subplots

raw_data_cache = 'nem_data'

# Build data set for calibrating the dispatch planner's price forecasting model.
start_time_historical_data = '2020/11/01 00:00:00'
end_time_historical_data = '2020/12/01 00:00:00'

price_data = historical_inputs.get_regional_prices(start_time_historical_data,
                                                   end_time_historical_data,
                                                   raw_data_cache)

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
historical_data['nsw-raise_regulation-fleet-dispatch'] = 0.0
historical_data['nsw-raise_6_seconds-fleet-dispatch'] = 0.0
historical_data['nsw-raise_60_seconds-fleet-dispatch'] = 0.0
historical_data['nsw-raise_5_minutes-fleet-dispatch'] = 0.0

# Build data set for running the dispatch planner's price forecasting model.
start_time_forward_data = '2021/01/01 00:00:00'
end_time_forward_data = '2021/01/07 00:00:00'

demand_data = historical_inputs.get_regional_demand(start_time_forward_data,
                                                    end_time_forward_data,
                                                    raw_data_cache)

tech_availability = historical_inputs.get_tech_operating_capacities(start_time_forward_data,
                                                                    end_time_forward_data,
                                                                    raw_data_cache)

forward_data = pd.merge(demand_data, tech_availability, on='SETTLEMENTDATE')
forward_data = forward_data[forward_data['SETTLEMENTDATE'].dt.hour.isin([0, 3, 6, 9, 12, 15, 18, 21])]
forward_data = forward_data.reset_index(drop=True)
forward_data['interval'] = forward_data.index
forward_data = forward_data.drop(columns=['SETTLEMENTDATE'])

p = planner.DispatchPlanner(dispatch_interval=60 * 3, historical_data=historical_data,
                            forward_data=forward_data, demand_delta_steps=50)

p.add_unit('thermal_one', 'nsw', initial_mw=310.0)
p.add_unit_to_market_flow('thermal_one', capacity=660)
p.add_generator('thermal_one', capacity=1000, cost=30)
p.add_unit_minimum_operating_level('thermal_one', min_loading=310, shutdown_ramp_rate=120, start_up_ramp_rate=120,
                                   min_down_time=60 * 8, min_up_time=60 * 8, initial_state=1,
                                   initial_up_time=60 * 24 * 7, initial_down_time=0)
p.add_startup_costs('thermal_one', hot_start_cost=120 * 1000, cold_start_cost=350 * 1000,
                    time_to_go_cold=60 * 24)
p.add_ramp_rates('thermal_one', ramp_up_rate=310, ramp_down_rate=230)
p.set_unit_fcas_region('thermal_one', 'raise_regulation', 'nsw')
p.set_unit_fcas_region('thermal_one', 'raise_6_seconds', 'nsw')
p.set_unit_fcas_region('thermal_one', 'raise_60_seconds', 'nsw')
p.set_unit_fcas_region('thermal_one', 'raise_5_minutes', 'nsw')
p.add_regulation_service_to_output('thermal_one', 'raise_regulation', availability=100, ramp_rate=240)
p.add_contingency_service_to_output('thermal_one', 'raise_6_seconds', availability=66)
p.add_contingency_service_to_output('thermal_one', 'raise_60_seconds', availability=66)
p.add_contingency_service_to_output('thermal_one', 'raise_5_minutes', availability=66)

p.add_regional_market('nsw', 'energy')
p.add_regional_market('nsw', 'raise_regulation')
p.add_regional_market('nsw', 'raise_6_seconds')
p.add_regional_market('nsw', 'raise_60_seconds')
p.add_regional_market('nsw', 'raise_5_minutes')

p.optimise()

dispatch = p.get_unit_energy_flows('thermal_one')

fcas_dispatch = p.get_fcas_dispatch('thermal_one')
fcas_dispatch.to_csv('fcas_dispatch.csv')

forecast = p.get_nominal_price_forecast('nsw', 'energy')

dispatch.to_csv('thermal_one_dispatch.csv')

price_data = historical_inputs.get_regional_prices(start_time_forward_data,
                                                   end_time_forward_data,
                                                   raw_data_cache)

hist_dispatch = historical_inputs.get_fleet_dispatch(start_time_forward_data,
                                                     end_time_forward_data,
                                                     ['BW01'],
                                                     raw_data_cache)

price_data = price_data[price_data['SETTLEMENTDATE'].dt.hour.isin([0, 3, 6, 9, 12, 15, 18, 21])]
price_data = price_data.reset_index(drop=True)
price_data['interval'] = price_data.index

hist_dispatch = hist_dispatch[hist_dispatch['SETTLEMENTDATE'].dt.hour.isin([0, 3, 6, 9, 12, 15, 18, 21])]
hist_dispatch = hist_dispatch.reset_index(drop=True)
hist_dispatch['interval'] = hist_dispatch.index

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=dispatch['interval'], y=dispatch['net_dispatch'], name='net dispatch'), secondary_y=False)
fig.add_trace(go.Scatter(x=hist_dispatch['interval'], y=hist_dispatch['fleet_dispatch'], name='hist dispatch'),
              secondary_y=False)
fig.add_trace(go.Scatter(x=price_data['interval'], y=price_data['nsw-energy'], name='price'), secondary_y=True)
fig.add_trace(go.Scatter(x=forecast['interval'], y=forecast[0], name='price forecast'), secondary_y=True)
fig.show()
