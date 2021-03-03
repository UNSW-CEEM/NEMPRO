from nemosis import data_fetch_methods
from nemosis import defaults
import pandas as pd

start_time = '2021/01/14 00:00:00'
end_time = '2021/01/21 00:00:00'

v_bids = data_fetch_methods.dynamic_data_compiler(start_time, end_time, 'BIDPEROFFER_D', 'nem_data',
                                                  filter_cols=['DUID'],
                                                  filter_values=[('BW01',)])
p_bids = data_fetch_methods.dynamic_data_compiler(start_time, end_time, 'BIDDAYOFFER_D', 'nem_data',
                                                  filter_cols=['DUID', 'BIDTYPE'],
                                                  filter_values=[('BW01',), ('ENERGY',)])
dispatch = data_fetch_methods.dynamic_data_compiler(start_time, end_time, 'DISPATCH_UNIT_SCADA', 'nem_data',
                                                    filter_cols=['DUID'], filter_values=[('BW01',)])

dispatch_targets = data_fetch_methods.dynamic_data_compiler(start_time, end_time, 'DISPATCHLOAD', 'nem_data',
                                                            filter_cols=['DUID'], filter_values=[('BW01',)])

v_bids.to_csv('v_bids.csv')
p_bids.to_csv('p_bids.csv')
dispatch.to_csv('dispatch.csv')
dispatch_targets.to_csv('dispatch_targets.csv')