import pandas as pd
from NEMPRO import historical_inputs, planner, units, residual_demand
import plotly.graph_objects as go
from plotly.subplots import make_subplots


raw_data_cache = 'C:/Users/nick/Documents/nem_data'

# Build data set for calibrating the dispatch planner's price forecasting model.
start_time_historical_data = '2019/01/01 00:00:00'
end_time_historical_data = '2020/01/01 00:00:00'

regions_short_names ={
    'QLD1': 'qld',
    'NSW1': 'nsw',
    'VIC1': 'vic',
    'SA1': 'sa',
    'TAS1': 'tas'
}

historical_data = residual_demand.get(start_time_historical_data, end_time_historical_data,  raw_data_cache)


for region in ['QLD1', 'NSW1', 'VIC1', 'SA1', 'TAS1']:
    frac_peak_demand = residual_demand.get_region_fraction_of_max_residual_demand(historical_data, region)
    regional_data = historical_data[historical_data['REGIONID'] == region].loc[:, ['SETTLEMENTDATE', 'RESIDUALDEMAND']]
    forward_data = regional_data.copy()

    r = regions_short_names[region]
    regional_data = regional_data.rename(columns={'RESIDUALDEMAND': '{}-demand'.format(r)})
    regional_data['{}-energy'.format(r)] = regional_data['{}-demand'.format(r)]
    regional_data['{}-energy-fleet-dispatch'.format(r)] = 0.0
    regional_data = regional_data.reset_index(drop=True)
    regional_data['interval'] = regional_data.index
    regional_data = regional_data.drop(columns=['SETTLEMENTDATE'])

    forward_data = forward_data.reset_index(drop=True)
    forward_data['interval'] = forward_data.index
    settlement_dates = forward_data.loc[:, ['interval', 'SETTLEMENTDATE', 'RESIDUALDEMAND']]
    forward_data = forward_data.rename(columns={'RESIDUALDEMAND': '{}-demand'.format(r)})
    forward_data = forward_data.drop(columns=['SETTLEMENTDATE'])

    for battery_capacity_mw in range(0, 3300, 300):
        for storage_hours in [0.5, 1, 2, 3, 4, 8]:

            p = planner.DispatchPlanner(dispatch_interval=5, historical_data=regional_data,
                                        forward_data=forward_data, demand_delta_steps=1000, train_pct=0.005)

            battery = units.GenericUnit(p, initial_dispatch=0.0, optimisation_time_step=5)
            battery.set_service_region('energy', r)
            battery.add_from_market_energy_flow(battery_capacity_mw)
            battery.add_to_market_energy_flow(battery_capacity_mw)
            battery.add_storage(mwh=battery_capacity_mw * storage_hours,
                                initial_mwh=battery_capacity_mw * storage_hours * 0.5,
                                output_capacity=battery_capacity_mw, input_capacity=battery_capacity_mw,
                                input_efficiency=0.9, output_efficiency=0.9)

            p.add_demand_smoothing_objective_function(r, 'energy')

            p.optimise()

            dispatch = battery.get_dispatch()

            dispatch = pd.merge(dispatch, settlement_dates, on='interval')
            dispatch.to_csv('battery_dispatch_profiles_v2/{}_{}_{}_month.csv'.format(r, battery_capacity_mw, storage_hours),
                            index=False)
