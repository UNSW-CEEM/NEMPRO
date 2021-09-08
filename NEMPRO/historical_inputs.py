from nemosis import data_fetch_methods, defaults
import pandas as pd

aemo_price_names = {'energy': 'RRP',
                    'raise_regulation': 'RAISEREGRRP',
                    'raise_6_second': 'RAISE6SECRRP',
                    'raise_60_second': 'RAISE60SECRRP',
                    'raise_5_minute': 'RAISE5MINRRP'}


def get_regional_prices(start_time, end_time, raw_data_cache):

    dispatch_data = data_fetch_methods.dynamic_data_compiler(start_time, end_time, 'DISPATCHPRICE', raw_data_cache,
                                                             select_columns=['SETTLEMENTDATE', 'INTERVENTION',
                                                                             'REGIONID', 'RRP', 'RAISEREGRRP',
                                                                             'RAISE6SECRRP', 'RAISE60SECRRP',
                                                                             'RAISE5MINRRP'])

    dispatch_data = dispatch_data[dispatch_data['INTERVENTION'] == 0]
    data = pd.DataFrame()
    for name, aemo_name in aemo_price_names.items():
        dispatch_data[aemo_name] = pd.to_numeric(dispatch_data[aemo_name])
        data_temp = dispatch_data.pivot_table(values=aemo_name, index='SETTLEMENTDATE', columns='REGIONID')
        data_temp = data_temp.reset_index().fillna('0.0')
        data_temp = data_temp.rename(columns={'QLD1': 'qld', 'NSW1': 'nsw', 'VIC1': 'vic', 'SA1': 'sa', 'TAS1': 'tas'})
        data_temp.columns = [col + '-' + name if col != 'SETTLEMENTDATE' else col for col in data_temp.columns]
        if data.empty:
            data = data_temp
        else:
            data = pd.merge(data, data_temp, on=['SETTLEMENTDATE'])

    return data


def get_regional_demand(start_time, end_time, raw_data_cache):

    dispatch_data = data_fetch_methods.dynamic_data_compiler(start_time, end_time, 'DISPATCHREGIONSUM', raw_data_cache,
                                                             select_columns=['SETTLEMENTDATE', 'INTERVENTION',
                                                                             'REGIONID', 'TOTALDEMAND'])

    dispatch_data = dispatch_data[dispatch_data['INTERVENTION'] == 0]

    dispatch_data['TOTALDEMAND'] = pd.to_numeric(dispatch_data['TOTALDEMAND'])

    dispatch_data = dispatch_data.pivot_table(values='TOTALDEMAND', index='SETTLEMENTDATE', columns='REGIONID')

    dispatch_data = dispatch_data.reset_index().fillna('0.0')

    dispatch_data = dispatch_data.rename(columns={'QLD1': 'qld', 'NSW1': 'nsw', 'VIC1': 'vic', 'SA1': 'sa',
                                                  'TAS1': 'tas'})

    dispatch_data.columns = [col + '-demand' if col != 'SETTLEMENTDATE' else col for col in dispatch_data.columns]

    return dispatch_data


def get_duid_techs(raw_data_cache):

    cols = ['DUID', 'Region', 'Fuel Source - Descriptor', 'Technology Type - Descriptor']
    tech_data = data_fetch_methods.static_table_xl('Generators and Scheduled Loads', raw_data_cache, select_columns=cols)

    def tech_classifier(fuel_source, technology_type):
        category = fuel_source
        if technology_type == 'Hydro - Gravity':
            category = 'Hydro'
        elif technology_type == 'Open Cycle Gas turbines (OCGT)':
            category = 'OCGT'
        elif technology_type == 'Combined Cycle Gas Turbine (CCGT)':
            category = 'CCGT'
        elif technology_type == 'Run of River' or fuel_source == 'Solar' or fuel_source == 'Wind' or fuel_source == 'Solar ':
            category = 'ZEROSRMC'
        elif technology_type == 'Spark Ignition Reciprocating Engine':
            category = 'Engine'
        elif technology_type == 'Compression Reciprocating Engine':
            category = 'Engine'
        elif technology_type == 'Steam Sub-Critical' and (fuel_source == 'Natural Gas / Fuel Oil' or fuel_source == 'Natural Gas'):
            category = 'Gas Thermal'
        elif technology_type == 'Pump Storage' or technology_type == 'Battery':
            category = 'Storage'
        return category

    tech_data['TECH'] = tech_data.apply(lambda x: tech_classifier(x['Fuel Source - Descriptor'],
                                                                  x['Technology Type - Descriptor']),
                                            axis=1)

    return tech_data.loc[:, ['DUID', 'Region', 'TECH']]


def get_tech_operating_capacities(start_time, end_time, raw_data_cache):
    tech_data = get_duid_techs(raw_data_cache)

    dispatch_data = data_fetch_methods.dynamic_data_compiler(start_time, end_time, 'DISPATCHLOAD', raw_data_cache,
                                                             select_columns=['DUID', 'SETTLEMENTDATE',
                                                                             'INTERVENTION', 'AVAILABILITY'])

    dispatch_data = dispatch_data[dispatch_data['INTERVENTION'] == 0]


    dispatch_data = pd.merge(dispatch_data, tech_data, on='DUID')

    dispatch_data['AVAILABILITY'] = pd.to_numeric(dispatch_data['AVAILABILITY'])

    dispatch_data = dispatch_data.groupby(['TECH', 'SETTLEMENTDATE'], as_index=False).aggregate({'AVAILABILITY': 'sum'})

    dispatch_data['tech_region'] = dispatch_data['TECH'] + '-capacity'

    dispatch_data = dispatch_data.pivot_table(values='AVAILABILITY', index='SETTLEMENTDATE', columns='tech_region')

    dispatch_data = dispatch_data.reset_index().fillna('0.0')

    return dispatch_data


def get_fleet_dispatch(start_time, end_time, fleet_units, region, raw_data_cache):

    dispatch_data = data_fetch_methods.dynamic_data_compiler(start_time, end_time, 'DISPATCHLOAD', raw_data_cache,
                                                             select_columns=['DUID', 'SETTLEMENTDATE', 'TOTALCLEARED',
                                                                             'INTERVENTION', 'RAISEREG', 'RAISE6SEC',
                                                                             'RAISE60SEC', 'RAISE5MIN'])
    dispatch_data = dispatch_data[dispatch_data['INTERVENTION'] == 0]

    dispatch_data = dispatch_data[dispatch_data['DUID'].isin(fleet_units)]

    dispatch_data['TOTALCLEARED'] = pd.to_numeric(dispatch_data['TOTALCLEARED'])
    dispatch_data['RAISEREG'] = pd.to_numeric(dispatch_data['RAISEREG'])
    dispatch_data['RAISE6SEC'] = pd.to_numeric(dispatch_data['RAISE6SEC'])
    dispatch_data['RAISE60SEC'] = pd.to_numeric(dispatch_data['RAISE60SEC'])
    dispatch_data['RAISE5MIN'] = pd.to_numeric(dispatch_data['RAISE5MIN'])

    dispatch_data = dispatch_data.groupby('SETTLEMENTDATE', as_index=False).aggregate(
        {'TOTALCLEARED': 'sum', 'RAISEREG': 'sum', 'RAISE6SEC': 'sum', 'RAISE60SEC': 'sum', 'RAISE5MIN': 'sum'})

    aemo_dispatch_names = {'TOTALCLEARED': region + '-energy-fleet-dispatch',
                           'RAISEREG': region + '-raise_regulation-fleet-dispatch',
                           'RAISE6SEC': region + '-raise_6_second-fleet-dispatch',
                           'RAISE60SEC': region + '-raise_60_second-fleet-dispatch',
                           'RAISE5MIN': region + '-raise_5_minute-fleet-dispatch'}

    dispatch_data = dispatch_data.rename(columns=aemo_dispatch_names)

    return dispatch_data


def get_unit_dispatch(start_time, end_time, unit, raw_data_cache):
    dispatch_data = data_fetch_methods.dynamic_data_compiler(start_time, end_time, 'DISPATCHLOAD', raw_data_cache,
                                                             select_columns=['DUID', 'SETTLEMENTDATE', 'INTERVENTION',
                                                                             'INITIALMW'])
    dispatch_data = dispatch_data[dispatch_data['INTERVENTION'] == 0]
    dispatch_data = dispatch_data[dispatch_data['DUID'] == unit]
    initial_mw = dispatch_data['INITIALMW'].iloc[0]
    return float(initial_mw)


def get_residual_demand(start_time, end_time, raw_data_cache):
    cols = ['DUID', 'Region', 'Fuel Source - Descriptor']
    tech_data = data_fetch_methods.static_table_xl('Generators and Scheduled Loads', raw_data_cache, select_columns=cols)
    zero_srmc_techs = ['Wind', 'Solar', 'Solar ']
    tech_data = tech_data[tech_data['Fuel Source - Descriptor'].isin(zero_srmc_techs)]
    scada_data = data_fetch_methods.dynamic_data_compiler(start_time, end_time, 'DISPATCH_UNIT_SCADA', raw_data_cache)
    scada_data = pd.merge(scada_data, tech_data, on='DUID')
    scada_data['SCADAVALUE'] = pd.to_numeric(scada_data['SCADAVALUE'])
    scada_data = scada_data.groupby(['SETTLEMENTDATE', 'Region'], as_index=False).agg({'SCADAVALUE': 'sum'})
    regional_demand = data_fetch_methods.dynamic_data_compiler(start_time, end_time, 'DISPATCHREGIONSUM', raw_data_cache)
    regional_demand = regional_demand[regional_demand['INTERVENTION'] == 0]
    regional_demand = pd.merge(regional_demand, scada_data, left_on=['SETTLEMENTDATE', 'REGIONID'],
                               right_on=['SETTLEMENTDATE', 'Region'])
    regional_demand['TOTALDEMAND'] = pd.to_numeric(regional_demand['TOTALDEMAND'])
    regional_demand['RESIDUALDEMAND'] = regional_demand['TOTALDEMAND'] - regional_demand['SCADAVALUE']

    regional_demand = regional_demand.pivot_table(values='RESIDUALDEMAND', index='SETTLEMENTDATE', columns='REGIONID')

    regional_demand = regional_demand.reset_index().fillna('0.0')

    regional_demand = regional_demand.rename(columns={'QLD1': 'qld', 'NSW1': 'nsw', 'VIC1': 'vic', 'SA1': 'sa',
                                                  'TAS1': 'tas'})

    regional_demand.columns = [col + '-demand' if col != 'SETTLEMENTDATE' else col for col in regional_demand.columns]
    return regional_demand


def get_region_fraction_of_max_residual_demand(regional_demand, region):
    regional_demand = regional_demand.groupby('REGIONID', as_index=False).agg({'RESIDUALDEMAND': 'max'})
    sum_regional_max_regional_demands = regional_demand['RESIDUALDEMAND'].max()
    regional_max_demand = regional_demand[regional_demand['REGIONID'] == region]['RESIDUALDEMAND'].iloc[0]
    return regional_max_demand / sum_regional_max_regional_demands