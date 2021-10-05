import pandas as pd
import numpy as np
from pandas._testing import assert_frame_equal
from NEMPRO import planner, units


def test_energy_storage_over_two_intervals_with_inelastic_prices():
    forward_data = pd.DataFrame({
        'interval': [0, 1],
        'nsw-energy': [100, 200]})

    p = planner.DispatchPlanner(dispatch_interval=60, planning_horizon=2)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(capacity=100.0)
    u.add_from_market_energy_flow(capacity=100.0)
    u.add_storage(mwh=100.0, initial_mwh=0.0, output_capacity=100.0, input_capacity=100.0,
                  output_efficiency=1.0, input_efficiency=1.0)

    p.add_regional_market('nsw', 'energy', forecast=forward_data)

    p.optimise()

    dispatch = u.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0, 1],
        'net_dispatch': [-100.0, 100.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_energy_storage_over_two_intervals_with_inelastic_prices_with_inefficiencies():
    forward_data = pd.DataFrame({
        'interval': [0, 1],
        'nsw-energy': [100, 200]})

    p = planner.DispatchPlanner(dispatch_interval=60, planning_horizon=2)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(capacity=100.0)
    u.add_from_market_energy_flow(capacity=100.0)
    u.add_storage(mwh=100.0, initial_mwh=0.0, output_capacity=100.0, input_capacity=100.0,
                  output_efficiency=0.8, input_efficiency=0.9)

    p.add_regional_market('nsw', 'energy', forecast=forward_data)

    p.optimise()

    dispatch = u.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0, 1],
        'net_dispatch': [-100.0, 100 * 0.9 * 0.8]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_energy_storage_over_three_intervals_with_elastic_prices():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-demand': np.linspace(0, 500, num=101)})

    forward_data = pd.DataFrame({
        'interval': [0, 1, 2],
        'nsw-demand': [100.0, 400.0, 400.0]})

    f = planner.Forecaster()
    f.train(historical_data, train_sample_fraction=1.0, target_col='nsw-energy')
    price_forecast = f.price_forecast_with_generation_sensitivities(
        forward_data, region='nsw', market='energy', min_delta=-50, max_delta=50, steps=100)

    p = planner.DispatchPlanner(dispatch_interval=60, planning_horizon=3)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(capacity=50.0)
    u.add_from_market_energy_flow(capacity=50.0)
    u.add_storage(mwh=50.0, initial_mwh=0.0, output_capacity=50.0, input_capacity=50.0,
                  output_efficiency=1.0, input_efficiency=1.0)

    p.add_regional_market('nsw', 'energy', forecast=price_forecast)

    p.optimise()

    dispatch = u.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0, 1, 2],
        'net_dispatch': [-50.0, 25.0, 25.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)
