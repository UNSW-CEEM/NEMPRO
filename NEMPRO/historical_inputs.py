from nemosis import data_fetch_methods
import pandas as pd


def get_regional_prices(start_time, end_time, raw_data_cache):

    dispatch_data = data_fetch_methods.dynamic_data_compiler(start_time, end_time, 'DISPATCHPRICE', raw_data_cache,
                                                             select_columns=['SETTLEMENTDATE', 'INTERVENTION',
                                                                             'REGIONID', 'RRP'])

    dispatch_data = dispatch_data[dispatch_data['INTERVENTION'] == '0']

    dispatch_data['RRP'] = pd.to_numeric(dispatch_data['RRP'])

    dispatch_data = dispatch_data.pivot_table(values='RRP', index='SETTLEMENTDATE', columns='REGIONID')

    dispatch_data = dispatch_data.reset_index().fillna('0.0')

    dispatch_data = dispatch_data.rename(columns={'QLD1': 'qld', 'NSW1': 'nsw', 'VIC1': 'vic', 'SA1': 'sa',
                                                  'TAS1': 'tas'})

    dispatch_data.columns = [col + '-energy' if col != 'SETTLEMENTDATE' else col for col in dispatch_data.columns]

    return dispatch_data


def get_regional_demand(start_time, end_time, raw_data_cache):

    dispatch_data = data_fetch_methods.dynamic_data_compiler(start_time, end_time, 'DISPATCHREGIONSUM', raw_data_cache,
                                                             select_columns=['SETTLEMENTDATE', 'INTERVENTION',
                                                                             'REGIONID', 'TOTALDEMAND'])

    dispatch_data = dispatch_data[dispatch_data['INTERVENTION'] == '0']

    dispatch_data['TOTALDEMAND'] = pd.to_numeric(dispatch_data['TOTALDEMAND'])

    dispatch_data = dispatch_data.pivot_table(values='TOTALDEMAND', index='SETTLEMENTDATE', columns='REGIONID')

    dispatch_data = dispatch_data.reset_index().fillna('0.0')

    dispatch_data = dispatch_data.rename(columns={'QLD1': 'qld', 'NSW1': 'nsw', 'VIC1': 'vic', 'SA1': 'sa',
                                                  'TAS1': 'tas'})

    dispatch_data.columns = [col + '-demand' if col != 'SETTLEMENTDATE' else col for col in dispatch_data.columns]

    return dispatch_data


def get_duid_techs(raw_data_cache):

    cols = ['DUID', 'Region', 'Fuel Source - Descriptor', 'Technology Type - Descriptor']
    tech_data = data_fetch_methods.static_table_xl('', '', 'Generators and Scheduled Loads', raw_data_cache,
                                                   select_columns=cols)

    def tech_classifier(fuel_source, technology_type):
        category = fuel_source
        if technology_type == 'Hydro - Gravity':
            category = 'Hydro'
        elif technology_type == 'Open Cycle Gas turbines (OCGT)':
            category = 'OCGT'
        elif technology_type == 'Combined Cycle Gas Turbine (CCGT)':
            category = 'CCGT'
        elif technology_type == 'Battery':
            category = 'Battery'
        elif technology_type == 'Run of River':
            category = 'Hydro'
        elif technology_type == 'Spark Ignition Reciprocating Engine':
            category = 'Engine'
        elif technology_type == 'Compression Reciprocating Engine':
            category = 'Engine'
        elif technology_type == 'Steam Sub-Critical' and (fuel_source == 'Natural Gas / Fuel Oil' or fuel_source == 'Natural Gas'):
            category = 'Gas Thermal'
        elif technology_type == 'Pump Storage':
            category = 'Hydro'
        return category

    tech_data['TECH'] = tech_data.apply(lambda x: tech_classifier(x['Fuel Source - Descriptor'],
                                                                      x['Technology Type - Descriptor']),
                                            axis=1)

    return tech_data.loc[:, ['DUID', 'Region', 'TECH']]


def get_tech_operating_capacities(start_time, end_time, raw_data_cache):
    dispatch_data = data_fetch_methods.dynamic_data_compiler(start_time, end_time, 'DISPATCHLOAD', raw_data_cache,
                                                             select_columns=['DUID', 'SETTLEMENTDATE',
                                                                             'INTERVENTION', 'AVAILABILITY'])

    dispatch_data = dispatch_data[dispatch_data['INTERVENTION'] == '0']

    tech_data = get_duid_techs(raw_data_cache)

    dispatch_data = pd.merge(dispatch_data, tech_data, on='DUID')

    dispatch_data['AVAILABILITY'] = pd.to_numeric(dispatch_data['AVAILABILITY'])

    dispatch_data = dispatch_data.groupby(['TECH', 'Region', 'SETTLEMENTDATE'], as_index=False).aggregate({'AVAILABILITY': 'sum'})

    dispatch_data['tech_region'] = dispatch_data['TECH'] + '-' + dispatch_data['Region'] + '-capacity'

    dispatch_data = dispatch_data.pivot_table(values='AVAILABILITY', index='SETTLEMENTDATE', columns='tech_region')

    dispatch_data = dispatch_data.reset_index().fillna('0.0')

    return dispatch_data


def get_fleet_dispatch(start_time, end_time, fleet_units, raw_data_cache):

    dispatch_data = data_fetch_methods.dynamic_data_compiler(start_time, end_time, 'DISPATCHLOAD', raw_data_cache,
                                                             select_columns=['DUID', 'SETTLEMENTDATE', 'TOTALCLEARED',
                                                                             'INTERVENTION'])
    dispatch_data = dispatch_data[dispatch_data['INTERVENTION'] == '0']

    dispatch_data = dispatch_data[dispatch_data['DUID'].isin(fleet_units)]

    dispatch_data = dispatch_data.groupby('SETTLEMENTDATE', as_index=False).aggregate({'TOTALCLEARED': 'sum'})

    dispatch_data = dispatch_data.rename(columns={'TOTALCLEARED': 'fleet_dispatch'})

    return dispatch_data