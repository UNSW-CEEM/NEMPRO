import pandas as pd

dispatch_data = pd.read_csv('binding_constraints_2020.csv')

dispatch_data = dispatch_data.groupby(['CONSTRAINTID', 'GENCONID_EFFECTIVEDATE', 'GENCONID_VERSIONNO'],
                                      as_index=False).size()

dispatch_data = dispatch_data.rename(columns={'size': 'count_binding'})

intervals_per_year = (365 * 24 * 12)

dispatch_data['pct_binding'] = dispatch_data['count_binding'] / intervals_per_year

dispatch_data.to_csv('constraint_binding_time_2020.csv')