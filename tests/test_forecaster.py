import pandas as pd
import numpy as np
from pandas._testing import assert_frame_equal
from NEMPRO import planner, units
from causalnex.structure.pytorch import DAGRegressor


def test_one_to_one_demand_to_price_relationship():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 500, num=500).astype(int),
        'nsw-energy': np.linspace(0, 500, num=500),
        'nsw-demand': np.linspace(0, 500, num=500)})

    forward_data = pd.DataFrame({
        'interval': [0, 1, 2],
        'nsw-demand': [0, 250, 500]})

    #f = planner.Forecaster()
    #f.train(historical_data, train_sample_fraction=1.0, target_col='nsw-energy')
    #price_forecast = f.single_trace_forecast(forward_data)

    reg = DAGRegressor(threshold=0.0,
                       alpha=0.0001,
                       beta=0.5,
                       fit_intercept=True,
                       hidden_layer_units=[10],
                       standardize=True
                       )

    X = historical_data.loc[:, ['nsw-demand']]
    y = historical_data['nsw-energy']

    reg.fit(X, y)

    blah = reg.predict(pd.DataFrame({'x': [0.0, 250.0, 500.0]}))
    # [2634.15699576 4040.53486056]

    x=1