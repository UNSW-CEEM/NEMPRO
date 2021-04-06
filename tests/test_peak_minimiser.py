import pandas as pd
import numpy as np
from pandas._testing import assert_frame_equal
from NEMPRO import planner, units


def test_energy_storage_over_two_intervals_with_inelastic_prices():
    historical_data = pd.DataFrame({
        'interval': [0, 1],
        'nsw-demand': [100, 200],
        'nsw-energy': [100, 200],
        'nsw-energy-fleet-dispatch': 0})

    forward_data = pd.DataFrame({
        'interval': [0, 1, 2],
        'nsw-demand': [100, 200, 190]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1, demand_delta_steps=500)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(capacity=100.0)
    u.add_from_market_energy_flow(capacity=100.0)
    u.add_storage(mwh=100.0, initial_mwh=100-63.3, output_capacity=100.0, input_capacity=100.0,
                  output_efficiency=1.0, input_efficiency=1.0)

    p.add_demand_smoothing_objective_function('nsw', 'energy')

    p.optimise()

    dispatch = u.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0, 1, 2],
        'net_dispatch': [-63.3, 36.7, 26.7]
    })

    assert_frame_equal(expect_dispatch, dispatch)