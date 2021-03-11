from nemosis import data_fetch_methods, defaults
import pandas as pd

start_time = '2020/01/01 00:00:00'
end_time = '2021/01/01 00:00:00'
raw_data_cache = 'C:/Users/nick/Documents/nem_data'

dispatch_data = data_fetch_methods.dynamic_data_compiler(start_time, end_time, 'DISPATCHCONSTRAINT', raw_data_cache)
dispatch_data['RHS'] = pd.to_numeric(dispatch_data['RHS'])
dispatch_data['LHS'] = pd.to_numeric(dispatch_data['LHS'])
dispatch_data = dispatch_data[(dispatch_data['RHS'] - dispatch_data['LHS']).abs() < 0.001]

dispatch_data.to_csv('binding_constraints_2020.csv')