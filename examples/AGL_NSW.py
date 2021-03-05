import pandas as pd
from NEMPRO import historical_inputs, planner, units
import plotly.graph_objects as go
from plotly.subplots import make_subplots

raw_data_cache = 'nem_data'

# Build data set for calibrating the dispatch planner's price forecasting model.
start_time_historical_data = '2020/08/01 00:00:00'
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

fleet_units = ['BW01', 'BW01', 'BW03', 'BW04', 'BROKENH1', 'HVGTS', 'LD01', 'LD02', 'LD03', 'LD04', 'NYNGAN1',
               'STWF1']

fleet_dispatch = historical_inputs.get_fleet_dispatch(start_time_historical_data, end_time_historical_data, fleet_units,
                                                      'nsw', raw_data_cache)

historical_data = pd.merge(price_data, demand_data, on='SETTLEMENTDATE')
historical_data = pd.merge(historical_data, tech_availability, on='SETTLEMENTDATE')
historical_data = pd.merge(historical_data, fleet_dispatch, on='SETTLEMENTDATE')
historical_data = historical_data.reset_index(drop=True)
historical_data['interval'] = historical_data.index
historical_data = historical_data.drop(columns=['SETTLEMENTDATE'])


# Build data set for running the dispatch planner's price forecasting model.
start_time_forward_data = '2021/01/01 00:00:00'
end_time_forward_data = '2021/01/14 00:00:00'

demand_data = historical_inputs.get_regional_demand(start_time_forward_data,
                                                    end_time_forward_data,
                                                    raw_data_cache)

tech_availability = historical_inputs.get_tech_operating_capacities(start_time_forward_data,
                                                                    end_time_forward_data,
                                                                    raw_data_cache)

forward_data = pd.merge(demand_data, tech_availability, on='SETTLEMENTDATE')
#forward_data = forward_data[forward_data['SETTLEMENTDATE'].dt.hour.isin([0, 3, 6, 9, 12, 15, 18, 21])]
forward_data = forward_data.reset_index(drop=True)
forward_data['interval'] = forward_data.index
forward_data = forward_data.drop(columns=['SETTLEMENTDATE'])

p = planner.DispatchPlanner(dispatch_interval=5, historical_data=historical_data,
                            forward_data=forward_data, demand_delta_steps=50)

initial_mw = historical_inputs.get_unit_dispatch('2020/12/31 23:55:00', '2021/01/01 00:00:00', 'BW01', raw_data_cache)

BW01 = units.GenericUnit(p, initial_dispatch=initial_mw, optimisation_time_step=int(60 * 4))
BW01.set_service_region('energy', 'nsw')
BW01.set_service_region('raise_regulation', 'nsw')
BW01.set_service_region('raise_6_second', 'nsw')
BW01.set_service_region('raise_60_second', 'nsw')
BW01.set_service_region('raise_5_minute', 'nsw')
BW01.add_to_market_energy_flow(660)
BW01.add_primary_energy_source(capacity=1000, cost=30)
BW01.add_unit_minimum_operating_level(min_loading=310.0, shutdown_ramp_rate=100.0/60, start_up_ramp_rate=120.0/60,
                                      min_up_time=60 * 8, min_down_time=60 * 8, time_in_initial_state=60 * 8)
BW01.add_startup_costs(hot_start_cost=120 * 1000, cold_start_cost=350 * 1000, time_to_go_cold=60 * 24)
BW01.add_ramp_rates_to_energy_flow_to_market(ramp_up_rate=310/60, ramp_down_rate=230/60)
BW01.add_regulation_service_to_output('raise_regulation', availability=100, ramp_rate=240/60)
BW01.add_contingency_service_to_output('raise_6_second', availability=66)
BW01.add_contingency_service_to_output('raise_60_second', availability=66)
BW01.add_contingency_service_to_output('raise_5_minute', availability=66)

initial_mw = historical_inputs.get_unit_dispatch('2020/12/31 23:55:00', '2021/01/01 00:00:00', 'BW02', raw_data_cache)
BW02 = units.GenericUnit(p, initial_dispatch=initial_mw, optimisation_time_step=int(60 * 4))
BW02.set_service_region('energy', 'nsw')
BW02.set_service_region('raise_regulation', 'nsw')
BW02.set_service_region('raise_6_second', 'nsw')
BW02.set_service_region('raise_60_second', 'nsw')
BW02.set_service_region('raise_5_minute', 'nsw')
BW02.add_to_market_energy_flow(660)
BW02.add_primary_energy_source(capacity=1000, cost=30)
BW02.add_unit_minimum_operating_level(min_loading=310.0, shutdown_ramp_rate=100.0/60, start_up_ramp_rate=120.0/60,
                                      min_up_time=60 * 8, min_down_time=60 * 8, time_in_initial_state=60 * 8)
BW02.add_startup_costs(hot_start_cost=120 * 1000, cold_start_cost=350 * 1000, time_to_go_cold=60 * 24)
BW02.add_ramp_rates_to_energy_flow_to_market(ramp_up_rate=310/60, ramp_down_rate=230/60)
BW02.add_regulation_service_to_output('raise_regulation', availability=100, ramp_rate=240/60)
BW02.add_contingency_service_to_output('raise_6_second', availability=66)
BW02.add_contingency_service_to_output('raise_60_second', availability=66)
BW02.add_contingency_service_to_output('raise_5_minute', availability=66)

initial_mw = historical_inputs.get_unit_dispatch('2020/12/31 23:55:00', '2021/01/01 00:00:00', 'BW03', raw_data_cache)
BW03 = units.GenericUnit(p, initial_dispatch=initial_mw, optimisation_time_step=int(60 * 4))
BW03.set_service_region('energy', 'nsw')
BW03.set_service_region('raise_regulation', 'nsw')
BW03.set_service_region('raise_6_second', 'nsw')
BW03.set_service_region('raise_60_second', 'nsw')
BW03.set_service_region('raise_5_minute', 'nsw')
BW03.add_to_market_energy_flow(660)
BW03.add_primary_energy_source(capacity=1000, cost=30)
BW03.add_unit_minimum_operating_level(min_loading=310.0, shutdown_ramp_rate=100.0/60, start_up_ramp_rate=120.0/60,
                                      min_up_time=60 * 8, min_down_time=60 * 8, time_in_initial_state=60 * 8)
BW03.add_startup_costs(hot_start_cost=120 * 1000, cold_start_cost=350 * 1000, time_to_go_cold=60 * 24)
BW03.add_ramp_rates_to_energy_flow_to_market(ramp_up_rate=310/60, ramp_down_rate=230/60)
BW03.add_regulation_service_to_output('raise_regulation', availability=100, ramp_rate=240/60)
BW03.add_contingency_service_to_output('raise_6_second', availability=66)
BW03.add_contingency_service_to_output('raise_60_second', availability=66)
BW03.add_contingency_service_to_output('raise_5_minute', availability=66)

initial_mw = historical_inputs.get_unit_dispatch('2020/12/31 23:55:00', '2021/01/01 00:00:00', 'BW04', raw_data_cache)
BW04 = units.GenericUnit(p, initial_dispatch=initial_mw, optimisation_time_step=int(60 * 4))
BW04.set_service_region('energy', 'nsw')
BW04.set_service_region('raise_regulation', 'nsw')
BW04.set_service_region('raise_6_second', 'nsw')
BW04.set_service_region('raise_60_second', 'nsw')
BW04.set_service_region('raise_5_minute', 'nsw')
BW04.add_to_market_energy_flow(660)
BW04.add_primary_energy_source(capacity=1000, cost=30)
BW04.add_unit_minimum_operating_level(min_loading=310.0, shutdown_ramp_rate=100.0/60, start_up_ramp_rate=120.0/60,
                                      min_up_time=60 * 8, min_down_time=60 * 8, time_in_initial_state=60 * 8)
BW04.add_startup_costs(hot_start_cost=120 * 1000, cold_start_cost=350 * 1000, time_to_go_cold=60 * 24)
BW04.add_ramp_rates_to_energy_flow_to_market(ramp_up_rate=310/60, ramp_down_rate=230/60)
BW04.add_regulation_service_to_output('raise_regulation', availability=100, ramp_rate=240/60)
BW04.add_contingency_service_to_output('raise_6_second', availability=66)
BW04.add_contingency_service_to_output('raise_60_second', availability=66)
BW04.add_contingency_service_to_output('raise_5_minute', availability=66)

initial_mw = historical_inputs.get_unit_dispatch('2020/12/31 23:55:00', '2021/01/01 00:00:00', 'LD01', raw_data_cache)
LD01 = units.GenericUnit(p, initial_dispatch=initial_mw, optimisation_time_step=int(60 * 4))
LD01.set_service_region('energy', 'nsw')
LD01.set_service_region('raise_regulation', 'nsw')
LD01.set_service_region('raise_6_second', 'nsw')
LD01.set_service_region('raise_60_second', 'nsw')
LD01.set_service_region('raise_5_minute', 'nsw')
LD01.add_to_market_energy_flow(500)
LD01.add_primary_energy_source(capacity=1000, cost=30)
LD01.add_unit_minimum_operating_level(min_loading=220.0, shutdown_ramp_rate=100.0/60, start_up_ramp_rate=120.0/60,
                                      min_up_time=60 * 8, min_down_time=60 * 8, time_in_initial_state=60 * 8)
LD01.add_startup_costs(hot_start_cost=120 * 1000, cold_start_cost=350 * 1000, time_to_go_cold=60 * 24)
LD01.add_ramp_rates_to_energy_flow_to_market(ramp_up_rate=240/60, ramp_down_rate=180/60)
LD01.add_regulation_service_to_output('raise_regulation', availability=1, ramp_rate=240/60)
LD01.add_contingency_service_to_output('raise_6_second', availability=30)
LD01.add_contingency_service_to_output('raise_60_second', availability=30)
LD01.add_contingency_service_to_output('raise_5_minute', availability=1)

initial_mw = historical_inputs.get_unit_dispatch('2020/12/31 23:55:00', '2021/01/01 00:00:00', 'LD02', raw_data_cache)
LD02 = units.GenericUnit(p, initial_dispatch=initial_mw, optimisation_time_step=int(60 * 4))
LD02.set_service_region('energy', 'nsw')
LD02.set_service_region('raise_regulation', 'nsw')
LD02.set_service_region('raise_6_second', 'nsw')
LD02.set_service_region('raise_60_second', 'nsw')
LD02.set_service_region('raise_5_minute', 'nsw')
LD02.add_to_market_energy_flow(500)
LD02.add_primary_energy_source(capacity=1000, cost=30)
LD02.add_unit_minimum_operating_level(min_loading=220.0, shutdown_ramp_rate=100.0/60, start_up_ramp_rate=120.0/60,
                                      min_up_time=60 * 8, min_down_time=60 * 8, time_in_initial_state=60 * 8)
LD02.add_startup_costs(hot_start_cost=120 * 1000, cold_start_cost=350 * 1000, time_to_go_cold=60 * 24)
LD02.add_ramp_rates_to_energy_flow_to_market(ramp_up_rate=240/60, ramp_down_rate=180/60)
LD02.add_regulation_service_to_output('raise_regulation', availability=1, ramp_rate=240/60)
LD02.add_contingency_service_to_output('raise_6_second', availability=30)
LD02.add_contingency_service_to_output('raise_60_second', availability=30)
LD02.add_contingency_service_to_output('raise_5_minute', availability=1)

initial_mw = historical_inputs.get_unit_dispatch('2020/12/31 23:55:00', '2021/01/01 00:00:00', 'LD04', raw_data_cache)
LD04 = units.GenericUnit(p, initial_dispatch=initial_mw, optimisation_time_step=int(60 * 4))
LD04.set_service_region('energy', 'nsw')
LD04.set_service_region('raise_regulation', 'nsw')
LD04.set_service_region('raise_6_second', 'nsw')
LD04.set_service_region('raise_60_second', 'nsw')
LD04.set_service_region('raise_5_minute', 'nsw')
LD04.add_to_market_energy_flow(500)
LD04.add_primary_energy_source(capacity=1000, cost=30)
LD04.add_unit_minimum_operating_level(min_loading=220.0, shutdown_ramp_rate=100.0/60, start_up_ramp_rate=120.0/60,
                                      min_up_time=60 * 8, min_down_time=60 * 8, time_in_initial_state=60 * 8)
LD04.add_startup_costs(hot_start_cost=120 * 1000, cold_start_cost=350 * 1000, time_to_go_cold=60 * 24)
LD04.add_ramp_rates_to_energy_flow_to_market(ramp_up_rate=240/60, ramp_down_rate=180/60)
LD04.add_regulation_service_to_output('raise_regulation', availability=1, ramp_rate=240/60)
LD04.add_contingency_service_to_output('raise_6_second', availability=30)
LD04.add_contingency_service_to_output('raise_60_second', availability=30)
LD04.add_contingency_service_to_output('raise_5_minute', availability=1)

p.add_regional_market('nsw', 'energy')
p.add_regional_market('nsw', 'raise_regulation')
p.add_regional_market('nsw', 'raise_6_second')
p.add_regional_market('nsw', 'raise_60_second')
p.add_regional_market('nsw', 'raise_5_minute')

p.optimise()

dispatch = BW01.get_dispatch()

# fcas_dispatch = p.get_fcas_dispatch('thermal_one')
# fcas_dispatch.to_csv('fcas_dispatch.csv')

forecast = p.get_nominal_price_forecast('nsw', 'energy')

dispatch.to_csv('thermal_one_dispatch.csv')

price_data = historical_inputs.get_regional_prices(start_time_forward_data,
                                                   end_time_forward_data,
                                                   raw_data_cache)

hist_dispatch = historical_inputs.get_fleet_dispatch(start_time_forward_data,
                                                     end_time_forward_data,
                                                     ['BW01'], 'nsw',
                                                     raw_data_cache)

#price_data = price_data[price_data['SETTLEMENTDATE'].dt.hour.isin([0, 3, 6, 9, 12, 15, 18, 21])]
price_data = price_data.reset_index(drop=True)
price_data['interval'] = price_data.index

#hist_dispatch = hist_dispatch[hist_dispatch['SETTLEMENTDATE'].dt.hour.isin([0, 3, 6, 9, 12, 15, 18, 21])]
hist_dispatch = hist_dispatch.reset_index(drop=True)
hist_dispatch['interval'] = hist_dispatch.index

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=dispatch['interval'], y=dispatch['net_dispatch'], name='net dispatch'), secondary_y=False)
fig.add_trace(go.Scatter(x=hist_dispatch['interval'], y=hist_dispatch['nsw-energy-fleet-dispatch'], name='hist dispatch'),
              secondary_y=False)
fig.add_trace(go.Scatter(x=price_data['interval'], y=price_data['nsw-energy'], name='price'), secondary_y=True)
fig.add_trace(go.Scatter(x=forecast['interval'], y=forecast[0], name='price forecast'), secondary_y=True)
fig.show()
