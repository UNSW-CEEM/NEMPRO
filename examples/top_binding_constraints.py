import pandas as pd
from nemosis import data_fetch_methods, defaults
import pandas as pd

start_time = '2020/01/01 00:00:00'
end_time = '2021/01/01 00:00:00'
raw_data_cache = 'C:/Users/nick/Documents/nem_data'

dispatch_data = pd.read_csv('binding_constraints_2020.csv')

dispatch_data = dispatch_data.groupby(['CONSTRAINTID', 'GENCONID_EFFECTIVEDATE', 'GENCONID_VERSIONNO'],
                                      as_index=False).size()

dispatch_data = dispatch_data.rename(columns={'size': 'count_binding'})

dispatch_data = dispatch_data[dispatch_data['count_binding'] >= 24 * 12]

dispatch_data.to_csv('constraint_binding_time_2020.csv')

constraint_rhs = data_fetch_methods.dynamic_data_compiler(start_time, end_time, 'GENERICCONSTRAINTRHS', raw_data_cache)

dispatch_data['GENCONID_EFFECTIVEDATE'] = pd.to_datetime(dispatch_data['GENCONID_EFFECTIVEDATE'])
dispatch_data['GENCONID_VERSIONNO'] = pd.to_numeric(dispatch_data['GENCONID_VERSIONNO'], downcast='integer').astype(str)

dispatch_data = pd.merge(dispatch_data, constraint_rhs, how='left',
                         left_on=['CONSTRAINTID', 'GENCONID_EFFECTIVEDATE',	'GENCONID_VERSIONNO'],
                         right_on=['GENCONID', 'EFFECTIVEDATE', 'VERSIONNO'])

dispatch_data.to_csv('primary_2020_cons_rhs_equations.csv')