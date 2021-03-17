from nemosis import data_fetch_methods
import pandas as pd


def get(start_time, end_time, raw_data_cache):
    cols = ['DUID', 'Region', 'Fuel Source - Descriptor']
    tech_data = data_fetch_methods.static_table_xl('Generators and Scheduled Loads', raw_data_cache, select_columns=cols)
    zero_srmc_techs = ['Wind', 'Solar', 'Solar ']
    tech_data = tech_data[tech_data['Fuel Source - Descriptor'].isin(zero_srmc_techs)]
    scada_data = data_fetch_methods.dynamic_data_compiler(start_time, end_time, 'DISPATCH_UNIT_SCADA', raw_data_cache)
    scada_data = pd.merge(scada_data, tech_data, on='DUID')
    scada_data['SCADAVALUE'] = pd.to_numeric(scada_data['SCADAVALUE'])
    scada_data = scada_data.groupby(['SETTLEMENTDATE', 'Region'], as_index=False).agg({'SCADAVALUE': 'sum'})
    regional_demand = data_fetch_methods.dynamic_data_compiler(start_time, end_time, 'DISPATCHREGIONSUM', raw_data_cache)
    regional_demand = pd.merge(regional_demand, scada_data, left_on=['SETTLEMENTDATE', 'REGIONID'],
                               right_on=['SETTLEMENTDATE', 'Region'])
    regional_demand['TOTALDEMAND'] = pd.to_numeric(regional_demand['TOTALDEMAND'])
    regional_demand['RESIDUALDEMAND'] = regional_demand['TOTALDEMAND'] - regional_demand['SCADAVALUE']
    return regional_demand.loc[:, ['SETTLEMENTDATE', 'REGIONID', 'RESIDUALDEMAND']]


def get_region_fraction_of_max_residual_demand(regional_demand, region):
    regional_demand = regional_demand.groupby('REGIONID', as_index=False).agg({'RESIDUALDEMAND': 'max'})
    sum_regional_max_regional_demands = regional_demand['RESIDUALDEMAND'].max()
    regional_max_demand = regional_demand[regional_demand['REGIONID'] == region]['RESIDUALDEMAND'].iloc[0]
    return regional_max_demand / sum_regional_max_regional_demands