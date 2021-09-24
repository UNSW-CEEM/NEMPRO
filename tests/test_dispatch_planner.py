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
        'nsw-demand': [100, 400, 400]})

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


def test_load_energy_and_raise_contingency_joint_capacity_con_lower_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-raise_60_second': np.linspace(0, 500, num=101),
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_60_second-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(25.0)
    u.set_service_region('raise_60_second', 'nsw')
    u.add_contingency_service_to_input('raise_60_second', availability=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_60_second')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [-25.0],
        'nsw-raise_60_second-dispatch': [25.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_load_energy_and_raise_contingency_joint_capacity_con_plateau():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-raise_60_second': np.linspace(0, 500, num=101),
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_60_second-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(45.0)
    u.set_service_region('raise_60_second', 'nsw')
    u.add_contingency_service_to_input('raise_60_second', availability=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_60_second')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [-45.0],
        'nsw-raise_60_second-dispatch': [40.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_load_energy_and_raise_contingency_joint_capacity_con_upper_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-raise_60_second': np.linspace(0, 500, num=101),
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_60_second-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(50.0)
    u.set_service_region('raise_60_second', 'nsw')
    u.add_contingency_service_to_input('raise_60_second', availability=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_60_second')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [-50.0],
        'nsw-raise_60_second-dispatch': [40.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_load_energy_and_raise_contingency_joint_capacity_con_explicit_trapezium_lower_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-raise_60_second': np.linspace(0, 500, num=101),
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_60_second-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(15.0)
    u.set_service_region('raise_60_second', 'nsw')

    fcas_trapezium = {'enablement_min': 10,
                      'low_breakpoint': 20,
                      'high_breakpoint': 30,
                      'enablement_max': 40}

    u.add_contingency_service_to_input('raise_60_second', availability=40.0, fcas_trapezium=fcas_trapezium)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_60_second')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [-15.0],
        'nsw-raise_60_second-dispatch': [20.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_load_energy_and_raise_contingency_joint_capacity_con_explicit_trapezium_plateau():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-raise_60_second': np.linspace(0, 500, num=101),
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_60_second-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(25.0)
    u.set_service_region('raise_60_second', 'nsw')

    fcas_trapezium = {'enablement_min': 10,
                      'low_breakpoint': 20,
                      'high_breakpoint': 30,
                      'enablement_max': 40}

    u.add_contingency_service_to_input('raise_60_second', availability=40.0, fcas_trapezium=fcas_trapezium)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_60_second')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [-25.0],
        'nsw-raise_60_second-dispatch': [40.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_load_energy_and_raise_contingency_joint_capacity_con_explicit_trapezium_upper_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-raise_60_second': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_60_second-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(35.0)
    u.set_service_region('raise_60_second', 'nsw')

    fcas_trapezium = {'enablement_min': 10,
                      'low_breakpoint': 20,
                      'high_breakpoint': 30,
                      'enablement_max': 40}

    u.add_contingency_service_to_input('raise_60_second', availability=40.0, fcas_trapezium=fcas_trapezium)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_60_second')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [-35.0],
        'nsw-raise_60_second-dispatch': [20.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_generator_energy_and_raise_contingency_joint_capacity_con_lower_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-raise_60_second': np.linspace(0, 500, num=101),
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_60_second-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(50.0)
    u.add_primary_energy_source(0.0)
    u.set_service_region('raise_60_second', 'nsw')
    u.add_contingency_service_to_output('raise_60_second', availability=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_60_second')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [0.0],
        'nsw-raise_60_second-dispatch': [40.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_generator_energy_and_raise_contingency_joint_capacity_con_plateau():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-raise_60_second': np.linspace(0, 500, num=101),
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_60_second-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(50.0)
    u.add_primary_energy_source(5.0)
    u.set_service_region('raise_60_second', 'nsw')
    u.add_contingency_service_to_output('raise_60_second', availability=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_60_second')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [5.0],
        'nsw-raise_60_second-dispatch': [40.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_generator_energy_and_raise_contingency_joint_capacity_con_upper_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-raise_60_second': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_60_second-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(50.0)
    u.add_primary_energy_source(30.0)
    u.set_service_region('raise_60_second', 'nsw')
    u.add_contingency_service_to_output('raise_60_second', availability=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_60_second')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [30.0],
        'nsw-raise_60_second-dispatch': [20.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_generator_energy_and_raise_contingency_joint_capacity_con_explicit_trapezium_lower_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-raise_60_second': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_60_second-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(50.0)
    u.add_primary_energy_source(15.0)
    u.set_service_region('raise_60_second', 'nsw')

    fcas_trapezium = {'enablement_min': 10,
                      'low_breakpoint': 20,
                      'high_breakpoint': 30,
                      'enablement_max': 40}

    u.add_contingency_service_to_output('raise_60_second', availability=40.0, fcas_trapezium=fcas_trapezium)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_60_second')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [15.0],
        'nsw-raise_60_second-dispatch': [20.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_generator_energy_and_raise_contingency_joint_capacity_con_explicit_trapezium_plateau():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-raise_60_second': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_60_second-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(50.0)
    u.add_primary_energy_source(20.0)
    u.set_service_region('raise_60_second', 'nsw')

    fcas_trapezium = {'enablement_min': 10,
                      'low_breakpoint': 20,
                      'high_breakpoint': 30,
                      'enablement_max': 40}

    u.add_contingency_service_to_output('raise_60_second', availability=40.0, fcas_trapezium=fcas_trapezium)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_60_second')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [20.0],
        'nsw-raise_60_second-dispatch': [40.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_generator_energy_and_raise_contingency_joint_capacity_con_explicit_trapezium_upper_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-raise_60_second': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_60_second-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(50.0)
    u.add_primary_energy_source(35.0)
    u.set_service_region('raise_60_second', 'nsw')

    fcas_trapezium = {'enablement_min': 10,
                      'low_breakpoint': 20,
                      'high_breakpoint': 30,
                      'enablement_max': 40}

    u.add_contingency_service_to_output('raise_60_second', availability=40.0, fcas_trapezium=fcas_trapezium)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_60_second')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [35.0],
        'nsw-raise_60_second-dispatch': [20.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_load_energy_and_lower_contingency_joint_capacity_con_lower_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-lower_60_second': np.linspace(0, 500, num=101),
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-lower_60_second-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(0.0)
    u.set_service_region('lower_60_second', 'nsw')
    u.add_contingency_service_to_input('lower_60_second', availability=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_60_second')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [0.0],
        'nsw-lower_60_second-dispatch': [40.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_load_energy_and_lower_contingency_joint_capacity_con_plateau():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-lower_60_second': np.linspace(0, 500, num=101),
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-lower_60_second-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(10.0)
    u.set_service_region('lower_60_second', 'nsw')
    u.add_contingency_service_to_input('lower_60_second', availability=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_60_second')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [-10.0],
        'nsw-lower_60_second-dispatch': [40.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_load_energy_and_lower_contingency_joint_capacity_con_upper_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-lower_60_second': np.linspace(0, 500, num=101),
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-lower_60_second-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(30.0)
    u.set_service_region('lower_60_second', 'nsw')
    u.add_contingency_service_to_input('lower_60_second', availability=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_60_second')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [-30.0],
        'nsw-lower_60_second-dispatch': [20.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_load_energy_and_lower_contingency_joint_capacity_con_explicit_trapezium_lower_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-lower_60_second': np.linspace(0, 500, num=101),
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-lower_60_second-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(15.0)
    u.set_service_region('lower_60_second', 'nsw')

    fcas_trapezium = {'enablement_min': 10,
                      'low_breakpoint': 20,
                      'high_breakpoint': 30,
                      'enablement_max': 40}

    u.add_contingency_service_to_input('lower_60_second', availability=40.0, fcas_trapezium=fcas_trapezium)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_60_second')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [-15.0],
        'nsw-lower_60_second-dispatch': [20.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_load_energy_and_lower_contingency_joint_capacity_con_explicit_trapezium_plateau():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-lower_60_second': np.linspace(0, 500, num=101),
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-lower_60_second-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(25.0)
    u.set_service_region('lower_60_second', 'nsw')

    fcas_trapezium = {'enablement_min': 10,
                      'low_breakpoint': 20,
                      'high_breakpoint': 30,
                      'enablement_max': 40}

    u.add_contingency_service_to_input('lower_60_second', availability=40.0,
                                       fcas_trapezium=fcas_trapezium)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_60_second')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [-25.0],
        'nsw-lower_60_second-dispatch': [40.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_load_energy_and_lower_contingency_joint_capacity_con_explicit_trapezium_upper_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-lower_60_second': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-lower_60_second-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(35.0)
    u.set_service_region('lower_60_second', 'nsw')

    fcas_trapezium = {'enablement_min': 10,
                      'low_breakpoint': 20,
                      'high_breakpoint': 30,
                      'enablement_max': 40}

    u.add_contingency_service_to_input('lower_60_second', availability=40.0, fcas_trapezium=fcas_trapezium)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_60_second')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [-35.0],
        'nsw-lower_60_second-dispatch': [20.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_generator_energy_and_lower_contingency_joint_capacity_con_lower_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-lower_60_second': np.linspace(0, 500, num=101),
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-lower_60_second-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(50.0)
    u.add_primary_energy_source(20.0)
    u.set_service_region('lower_60_second', 'nsw')
    u.add_contingency_service_to_output('lower_60_second', availability=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_60_second')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [20.0],
        'nsw-lower_60_second-dispatch': [20.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_generator_energy_and_lower_contingency_joint_capacity_con_plateau():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-lower_60_second': np.linspace(0, 500, num=101),
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-lower_60_second-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(50.0)
    u.add_primary_energy_source(45.0)
    u.set_service_region('lower_60_second', 'nsw')
    u.add_contingency_service_to_output('lower_60_second', availability=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_60_second')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [45.0],
        'nsw-lower_60_second-dispatch': [40.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_generator_energy_and_lower_contingency_joint_capacity_con_upper_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-lower_60_second': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-lower_60_second-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(50.0)
    u.add_primary_energy_source(50.0)
    u.set_service_region('lower_60_second', 'nsw')
    u.add_contingency_service_to_output('lower_60_second', availability=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_60_second')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [50.0],
        'nsw-lower_60_second-dispatch': [40.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_generator_energy_and_lower_contingency_joint_capacity_con_explicit_trapezium_lower_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-lower_60_second': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-lower_60_second-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(50.0)
    u.add_primary_energy_source(15.0)
    u.set_service_region('lower_60_second', 'nsw')

    fcas_trapezium = {'enablement_min': 10,
                      'low_breakpoint': 20,
                      'high_breakpoint': 30,
                      'enablement_max': 40}

    u.add_contingency_service_to_output('lower_60_second', availability=40.0, fcas_trapezium=fcas_trapezium)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_60_second')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [15.0],
        'nsw-lower_60_second-dispatch': [20.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_generator_energy_and_lower_contingency_joint_capacity_con_explicit_trapezium_plateau():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-lower_60_second': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-lower_60_second-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(50.0)
    u.add_primary_energy_source(20.0)
    u.set_service_region('lower_60_second', 'nsw')

    fcas_trapezium = {'enablement_min': 10,
                      'low_breakpoint': 20,
                      'high_breakpoint': 30,
                      'enablement_max': 40}

    u.add_contingency_service_to_output('lower_60_second', availability=40.0, fcas_trapezium=fcas_trapezium)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_60_second')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [20.0],
        'nsw-lower_60_second-dispatch': [40.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_load_energy_and_raise_reg_joint_capacity_con_lower_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-raise_regulation': np.linspace(0, 500, num=101),
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(25.0)
    u.set_service_region('raise_regulation', 'nsw')
    u.add_regulation_service_to_input('raise_regulation', availability=40.0, ramp_rate=60.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [-25.0],
        'nsw-raise_regulation-dispatch': [25.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_load_energy_and_raise_reg_joint_capacity_con_plateau():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-raise_regulation': np.linspace(0, 500, num=101),
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(45.0)
    u.set_service_region('raise_regulation', 'nsw')
    u.add_regulation_service_to_input('raise_regulation', availability=40.0, ramp_rate=60.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [-45.0],
        'nsw-raise_regulation-dispatch': [40.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_load_energy_and_raise_reg_joint_capacity_con_upper_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-raise_regulation': np.linspace(0, 500, num=101),
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(50.0)
    u.set_service_region('raise_regulation', 'nsw')
    u.add_regulation_service_to_input('raise_regulation', availability=40.0, ramp_rate=60.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [-50.0],
        'nsw-raise_regulation-dispatch': [40.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_load_energy_and_raise_reg_joint_capacity_con_explicit_trapezium_lower_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-raise_regulation': np.linspace(0, 500, num=101),
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(15.0)
    u.set_service_region('raise_regulation', 'nsw')

    fcas_trapezium = {'enablement_min': 10,
                      'low_breakpoint': 20,
                      'high_breakpoint': 30,
                      'enablement_max': 40}

    u.add_regulation_service_to_input('raise_regulation', availability=40.0, ramp_rate=60.0,
                                      fcas_trapezium=fcas_trapezium)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [-15.0],
        'nsw-raise_regulation-dispatch': [20.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_load_energy_and_raise_reg_joint_capacity_con_explicit_trapezium_plateau():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-raise_regulation': np.linspace(0, 500, num=101),
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(25.0)
    u.set_service_region('raise_regulation', 'nsw')

    fcas_trapezium = {'enablement_min': 10,
                      'low_breakpoint': 20,
                      'high_breakpoint': 30,
                      'enablement_max': 40}

    u.add_regulation_service_to_input('raise_regulation', availability=40.0, ramp_rate=60.0,
                                      fcas_trapezium=fcas_trapezium)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [-25.0],
        'nsw-raise_regulation-dispatch': [40.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_load_energy_and_raise_reg_joint_capacity_con_explicit_trapezium_upper_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-raise_regulation': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(35.0)
    u.set_service_region('raise_regulation', 'nsw')

    fcas_trapezium = {'enablement_min': 10,
                      'low_breakpoint': 20,
                      'high_breakpoint': 30,
                      'enablement_max': 40}

    u.add_regulation_service_to_input('raise_regulation', availability=40.0, ramp_rate=80.0,
                                      fcas_trapezium=fcas_trapezium)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [-35.0],
        'nsw-raise_regulation-dispatch': [20.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_generator_energy_and_raise_reg_joint_capacity_con_lower_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-raise_regulation': np.linspace(0, 500, num=101),
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(50.0)
    u.add_primary_energy_source(0.0)
    u.set_service_region('raise_regulation', 'nsw')
    u.add_regulation_service_to_output('raise_regulation', availability=40.0, ramp_rate=60.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [0.0],
        'nsw-raise_regulation-dispatch': [40.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_generator_energy_and_raise_reg_joint_capacity_con_plateau():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-raise_regulation': np.linspace(0, 500, num=101),
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(50.0)
    u.add_primary_energy_source(5.0)
    u.set_service_region('raise_regulation', 'nsw')
    u.add_regulation_service_to_output('raise_regulation', availability=40.0, ramp_rate=60.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [5.0],
        'nsw-raise_regulation-dispatch': [40.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_generator_energy_and_raise_reg_joint_capacity_con_upper_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-raise_regulation': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(50.0)
    u.add_primary_energy_source(30.0)
    u.set_service_region('raise_regulation', 'nsw')
    u.add_regulation_service_to_output('raise_regulation', availability=40.0, ramp_rate=60.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [30.0],
        'nsw-raise_regulation-dispatch': [20.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_generator_energy_and_raise_reg_joint_capacity_con_explicit_trapezium_lower_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-raise_regulation': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(50.0)
    u.add_primary_energy_source(15.0)
    u.set_service_region('raise_regulation', 'nsw')

    fcas_trapezium = {'enablement_min': 10,
                      'low_breakpoint': 20,
                      'high_breakpoint': 30,
                      'enablement_max': 40}

    u.add_regulation_service_to_output('raise_regulation', availability=40.0, ramp_rate=60.0,
                                       fcas_trapezium=fcas_trapezium)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [15.0],
        'nsw-raise_regulation-dispatch': [20.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_generator_energy_and_raise_reg_joint_capacity_con_explicit_trapezium_plateau():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-raise_regulation': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(50.0)
    u.add_primary_energy_source(20.0)
    u.set_service_region('raise_regulation', 'nsw')

    fcas_trapezium = {'enablement_min': 10,
                      'low_breakpoint': 20,
                      'high_breakpoint': 30,
                      'enablement_max': 40}

    u.add_regulation_service_to_output('raise_regulation', availability=40.0, ramp_rate=60.0,
                                       fcas_trapezium=fcas_trapezium)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [20.0],
        'nsw-raise_regulation-dispatch': [40.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_generator_energy_and_raise_reg_joint_capacity_con_explicit_trapezium_upper_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-raise_regulation': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(50.0)
    u.add_primary_energy_source(35.0)
    u.set_service_region('raise_regulation', 'nsw')

    fcas_trapezium = {'enablement_min': 10,
                      'low_breakpoint': 20,
                      'high_breakpoint': 30,
                      'enablement_max': 40}

    u.add_regulation_service_to_output('raise_regulation', availability=40.0, ramp_rate=60.0,
                                       fcas_trapezium=fcas_trapezium)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [35.0],
        'nsw-raise_regulation-dispatch': [20.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_load_energy_and_lower_reg_joint_capacity_con_lower_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-lower_regulation': np.linspace(0, 500, num=101),
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-lower_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(0.0)
    u.set_service_region('lower_regulation', 'nsw')
    u.add_regulation_service_to_input('lower_regulation', availability=40.0, ramp_rate=60.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [0.0],
        'nsw-lower_regulation-dispatch': [40.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_load_energy_and_lower_reg_joint_capacity_con_plateau():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-lower_regulation': np.linspace(0, 500, num=101),
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-lower_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(10.0)
    u.set_service_region('lower_regulation', 'nsw')
    u.add_regulation_service_to_input('lower_regulation', availability=40.0, ramp_rate=60.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [-10.0],
        'nsw-lower_regulation-dispatch': [40.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_load_energy_and_lower_reg_joint_capacity_con_upper_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-lower_regulation': np.linspace(0, 500, num=101),
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-lower_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(30.0)
    u.set_service_region('lower_regulation', 'nsw')
    u.add_regulation_service_to_input('lower_regulation', availability=40.0, ramp_rate=60.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [-30.0],
        'nsw-lower_regulation-dispatch': [20.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_load_energy_and_lower_reg_joint_capacity_con_explicit_trapezium_lower_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-lower_regulation': np.linspace(0, 500, num=101),
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-lower_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(15.0)
    u.set_service_region('lower_regulation', 'nsw')

    fcas_trapezium = {'enablement_min': 10,
                      'low_breakpoint': 20,
                      'high_breakpoint': 30,
                      'enablement_max': 40}

    u.add_regulation_service_to_input('lower_regulation', availability=40.0, ramp_rate=60.0,
                                      fcas_trapezium=fcas_trapezium)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [-15.0],
        'nsw-lower_regulation-dispatch': [20.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_load_energy_and_lower_reg_joint_capacity_con_explicit_trapezium_plateau():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-lower_regulation': np.linspace(0, 500, num=101),
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-lower_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(25.0)
    u.set_service_region('lower_regulation', 'nsw')

    fcas_trapezium = {'enablement_min': 10,
                      'low_breakpoint': 20,
                      'high_breakpoint': 30,
                      'enablement_max': 40}

    u.add_regulation_service_to_input('lower_regulation', availability=40.0, ramp_rate=80.0,
                                      fcas_trapezium=fcas_trapezium)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [-25.0],
        'nsw-lower_regulation-dispatch': [40.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_load_energy_and_lower_reg_joint_capacity_con_explicit_trapezium_upper_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-lower_regulation': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-lower_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(35.0)
    u.set_service_region('lower_regulation', 'nsw')

    fcas_trapezium = {'enablement_min': 10,
                      'low_breakpoint': 20,
                      'high_breakpoint': 30,
                      'enablement_max': 40}

    u.add_regulation_service_to_input('lower_regulation', availability=40.0, ramp_rate=80.0,
                                      fcas_trapezium=fcas_trapezium)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [-35.0],
        'nsw-lower_regulation-dispatch': [20.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_generator_energy_and_lower_reg_joint_capacity_con_lower_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-lower_regulation': np.linspace(0, 500, num=101),
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-lower_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(50.0)
    u.add_primary_energy_source(20.0)
    u.set_service_region('lower_regulation', 'nsw')
    u.add_regulation_service_to_output('lower_regulation', availability=40.0, ramp_rate=60.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [20.0],
        'nsw-lower_regulation-dispatch': [20.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_generator_energy_and_lower_reg_joint_capacity_con_plateau():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-lower_regulation': np.linspace(0, 500, num=101),
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-lower_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(50.0)
    u.add_primary_energy_source(45.0)
    u.set_service_region('lower_regulation', 'nsw')
    u.add_regulation_service_to_output('lower_regulation', availability=40.0, ramp_rate=60.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [45.0],
        'nsw-lower_regulation-dispatch': [40.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_generator_energy_and_lower_reg_joint_capacity_con_upper_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-lower_regulation': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-lower_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(50.0)
    u.add_primary_energy_source(50.0)
    u.set_service_region('lower_regulation', 'nsw')
    u.add_regulation_service_to_output('lower_regulation', availability=40.0, ramp_rate=60.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [50.0],
        'nsw-lower_regulation-dispatch': [40.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_generator_energy_and_lower_reg_joint_capacity_con_explicit_trapezium_lower_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-lower_regulation': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-lower_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(50.0)
    u.add_primary_energy_source(15.0)
    u.set_service_region('lower_regulation', 'nsw')

    fcas_trapezium = {'enablement_min': 10,
                      'low_breakpoint': 20,
                      'high_breakpoint': 30,
                      'enablement_max': 40}

    u.add_regulation_service_to_output('lower_regulation', availability=40.0, ramp_rate=60.0,
                                       fcas_trapezium=fcas_trapezium)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [15.0],
        'nsw-lower_regulation-dispatch': [20.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_generator_energy_and_lower_reg_joint_capacity_con_explicit_trapezium_plateau():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-lower_regulation': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-lower_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(50.0)
    u.add_primary_energy_source(20.0)
    u.set_service_region('lower_regulation', 'nsw')

    fcas_trapezium = {'enablement_min': 10,
                      'low_breakpoint': 20,
                      'high_breakpoint': 30,
                      'enablement_max': 40}

    u.add_regulation_service_to_output('lower_regulation', availability=40.0, ramp_rate=60.0,
                                       fcas_trapezium=fcas_trapezium)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [20.0],
        'nsw-lower_regulation-dispatch': [40.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_generator_energy_and_lower_reg_joint_capacity_con_explicit_trapezium_upper_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-lower_regulation': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-lower_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(50.0)
    u.add_primary_energy_source(35.0)
    u.set_service_region('lower_regulation', 'nsw')

    fcas_trapezium = {'enablement_min': 10,
                      'low_breakpoint': 20,
                      'high_breakpoint': 30,
                      'enablement_max': 40}

    u.add_regulation_service_to_output('lower_regulation', availability=40.0, ramp_rate=60.0,
                                       fcas_trapezium=fcas_trapezium)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [35.0],
        'nsw-lower_regulation-dispatch': [20.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_load_energy_and_raise_contingency_and_raise_reg_joint_capacity_con_lower_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-raise_regulation': np.linspace(0, 500, num=101),
        'nsw-raise_60_second': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_60_second-fleet-dispatch': np.zeros(101),
        'nsw-raise_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(25.0)

    u.set_service_region('raise_regulation', 'nsw')
    u.add_regulation_service_to_input('raise_regulation', availability=1.0, ramp_rate=60.0)

    u.set_service_region('raise_60_second', 'nsw')
    u.add_contingency_service_to_input('raise_60_second', availability=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_60_second')
    p.add_regional_market('nsw', 'raise_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [-25.0],
        'nsw-raise_60_second-dispatch': [24.0],
        'nsw-raise_regulation-dispatch': [1.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_load_energy_and_raise_contingency_raise_regulation_joint_capacity_con_plateau():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-raise_regulation': np.linspace(0, 500, num=101),
        'nsw-raise_60_second': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_60_second-fleet-dispatch': np.zeros(101),
        'nsw-raise_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(45.0)

    u.set_service_region('raise_regulation', 'nsw')
    u.add_regulation_service_to_input('raise_regulation', availability=1.0, ramp_rate=60.0)

    u.set_service_region('raise_60_second', 'nsw')
    u.add_contingency_service_to_input('raise_60_second', availability=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_60_second')
    p.add_regional_market('nsw', 'raise_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [-45.0],
        'nsw-raise_60_second-dispatch': [40.0],
        'nsw-raise_regulation-dispatch': [1.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_load_energy_and_raise_contingency_and_raise_regulation_joint_capacity_con_upper_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-raise_regulation': np.linspace(0, 500, num=101),
        'nsw-raise_60_second': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_60_second-fleet-dispatch': np.zeros(101),
        'nsw-raise_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(50.0)

    u.set_service_region('raise_regulation', 'nsw')
    u.add_regulation_service_to_input('raise_regulation', availability=1.0, ramp_rate=60.0)

    u.set_service_region('raise_60_second', 'nsw')
    u.add_contingency_service_to_input('raise_60_second', availability=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_60_second')
    p.add_regional_market('nsw', 'raise_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [-50.0],
        'nsw-raise_60_second-dispatch': [40.0],
        'nsw-raise_regulation-dispatch': [1.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_generator_energy_and_raise_contingency_and_raise_regulation_joint_capacity_con_lower_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-raise_regulation': np.linspace(0, 500, num=101),
        'nsw-raise_60_second': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_60_second-fleet-dispatch': np.zeros(101),
        'nsw-raise_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(50.0)
    u.add_primary_energy_source(0.0)

    u.set_service_region('raise_regulation', 'nsw')
    u.add_regulation_service_to_output('raise_regulation', availability=1.0, ramp_rate=60.0)

    u.set_service_region('raise_60_second', 'nsw')
    u.add_contingency_service_to_output('raise_60_second', availability=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_60_second')
    p.add_regional_market('nsw', 'raise_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [0.0],
        'nsw-raise_60_second-dispatch': [40.0],
        'nsw-raise_regulation-dispatch': [1.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_generator_energy_and_raise_contingency_and_raise_regulation_joint_capacity_con_plateau():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-raise_regulation': np.linspace(0, 500, num=101),
        'nsw-raise_60_second': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_60_second-fleet-dispatch': np.zeros(101),
        'nsw-raise_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(50.0)
    u.add_primary_energy_source(5.0)

    u.set_service_region('raise_regulation', 'nsw')
    u.add_regulation_service_to_output('raise_regulation', availability=1.0, ramp_rate=60.0)

    u.set_service_region('raise_60_second', 'nsw')
    u.add_contingency_service_to_output('raise_60_second', availability=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_60_second')
    p.add_regional_market('nsw', 'raise_regulation')

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_60_second')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [5.0],
        'nsw-raise_60_second-dispatch': [40.0],
        'nsw-raise_regulation-dispatch': [1.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_generator_energy_and_raise_contingency_and_raise_regulation_joint_capacity_con_upper_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-raise_regulation': np.linspace(0, 500, num=101),
        'nsw-raise_60_second': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_60_second-fleet-dispatch': np.zeros(101),
        'nsw-raise_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(50.0)
    u.add_primary_energy_source(30.0)

    u.set_service_region('raise_regulation', 'nsw')
    u.add_regulation_service_to_output('raise_regulation', availability=1.0, ramp_rate=60.0)

    u.set_service_region('raise_60_second', 'nsw')
    u.add_contingency_service_to_output('raise_60_second', availability=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_60_second')
    p.add_regional_market('nsw', 'raise_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [30.0],
        'nsw-raise_60_second-dispatch': [19.0],
        'nsw-raise_regulation-dispatch': [1.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_load_energy_and_raise_contingency_and_lower_reg_joint_capacity_con_lower_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-lower_regulation': np.linspace(0, 500, num=101),
        'nsw-raise_60_second': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_60_second-fleet-dispatch': np.zeros(101),
        'nsw-lower_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(25.0)

    u.set_service_region('lower_regulation', 'nsw')
    u.add_regulation_service_to_input('lower_regulation', availability=1.0, ramp_rate=60.0)

    u.set_service_region('raise_60_second', 'nsw')
    u.add_contingency_service_to_input('raise_60_second', availability=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_60_second')
    p.add_regional_market('nsw', 'lower_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [-25.0],
        'nsw-raise_60_second-dispatch': [25.0],
        'nsw-lower_regulation-dispatch': [1.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_load_energy_and_raise_contingency_lower_regulation_joint_capacity_con_plateau():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-lower_regulation': np.linspace(0, 500, num=101),
        'nsw-raise_60_second': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_60_second-fleet-dispatch': np.zeros(101),
        'nsw-lower_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(45.0)

    u.set_service_region('lower_regulation', 'nsw')
    u.add_regulation_service_to_input('lower_regulation', availability=1.0, ramp_rate=60.0)

    u.set_service_region('raise_60_second', 'nsw')
    u.add_contingency_service_to_input('raise_60_second', availability=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_60_second')
    p.add_regional_market('nsw', 'lower_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [-45.0],
        'nsw-raise_60_second-dispatch': [40.0],
        'nsw-lower_regulation-dispatch': [1.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_load_energy_and_raise_contingency_and_lower_regulation_joint_capacity_con_upper_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-lower_regulation': np.linspace(0, 500, num=101) * 10,
        'nsw-raise_60_second': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_60_second-fleet-dispatch': np.zeros(101),
        'nsw-lower_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(50.0)

    u.set_service_region('lower_regulation', 'nsw')
    u.add_regulation_service_to_input('lower_regulation', availability=1.0, ramp_rate=60.0)

    u.set_service_region('raise_60_second', 'nsw')
    u.add_contingency_service_to_input('raise_60_second', availability=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_60_second')
    p.add_regional_market('nsw', 'lower_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [-49.0],
        'nsw-raise_60_second-dispatch': [40.0],
        'nsw-lower_regulation-dispatch': [1.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_generator_energy_and_raise_contingency_and_lower_regulation_joint_capacity_con_lower_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-lower_regulation': np.linspace(0, 500, num=101),
        'nsw-raise_60_second': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_60_second-fleet-dispatch': np.zeros(101),
        'nsw-lower_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(50.0)
    u.add_primary_energy_source(0.0)

    u.set_service_region('lower_regulation', 'nsw')
    u.add_regulation_service_to_output('lower_regulation', availability=1.0, ramp_rate=60.0)

    u.set_service_region('raise_60_second', 'nsw')
    u.add_contingency_service_to_output('raise_60_second', availability=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_60_second')
    p.add_regional_market('nsw', 'lower_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [0.0],
        'nsw-raise_60_second-dispatch': [40.0],
        'nsw-lower_regulation-dispatch': [0.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_generator_energy_and_raise_contingency_and_lower_regulation_joint_capacity_con_plateau():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-lower_regulation': np.linspace(0, 500, num=101),
        'nsw-raise_60_second': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_60_second-fleet-dispatch': np.zeros(101),
        'nsw-lower_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(50.0)
    u.add_primary_energy_source(5.0)

    u.set_service_region('lower_regulation', 'nsw')
    u.add_regulation_service_to_output('lower_regulation', availability=1.0, ramp_rate=60.0)

    u.set_service_region('raise_60_second', 'nsw')
    u.add_contingency_service_to_output('raise_60_second', availability=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_60_second')
    p.add_regional_market('nsw', 'lower_regulation')

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_60_second')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [5.0],
        'nsw-raise_60_second-dispatch': [40.0],
        'nsw-lower_regulation-dispatch': [1.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_generator_energy_and_raise_contingency_and_lower_regulation_joint_capacity_con_upper_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-lower_regulation': np.linspace(0, 500, num=101),
        'nsw-raise_60_second': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_60_second-fleet-dispatch': np.zeros(101),
        'nsw-lower_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(50.0)
    u.add_primary_energy_source(30.0)

    u.set_service_region('lower_regulation', 'nsw')
    u.add_regulation_service_to_output('lower_regulation', availability=1.0, ramp_rate=60.0)

    u.set_service_region('raise_60_second', 'nsw')
    u.add_contingency_service_to_output('raise_60_second', availability=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_60_second')
    p.add_regional_market('nsw', 'lower_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [30.0],
        'nsw-raise_60_second-dispatch': [20.0],
        'nsw-lower_regulation-dispatch': [1.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_load_energy_and_lower_contingency_and_raise_reg_joint_capacity_con_lower_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-raise_regulation': np.linspace(0, 500, num=101),
        'nsw-lower_60_second': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-lower_60_second-fleet-dispatch': np.zeros(101),
        'nsw-raise_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(0.0)

    u.set_service_region('raise_regulation', 'nsw')
    u.add_regulation_service_to_input('raise_regulation', availability=1.0, ramp_rate=60.0)

    u.set_service_region('lower_60_second', 'nsw')
    u.add_contingency_service_to_input('lower_60_second', availability=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_60_second')
    p.add_regional_market('nsw', 'raise_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [0.0],
        'nsw-lower_60_second-dispatch': [40.0],
        'nsw-raise_regulation-dispatch': [0.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_load_energy_and_lower_contingency_raise_regulation_joint_capacity_con_plateau():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-raise_regulation': np.linspace(0, 500, num=101),
        'nsw-lower_60_second': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-lower_60_second-fleet-dispatch': np.zeros(101),
        'nsw-raise_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(5.0)

    u.set_service_region('raise_regulation', 'nsw')
    u.add_regulation_service_to_input('raise_regulation', availability=1.0, ramp_rate=60.0)

    u.set_service_region('lower_60_second', 'nsw')
    u.add_contingency_service_to_input('lower_60_second', availability=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_60_second')
    p.add_regional_market('nsw', 'raise_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [-5.0],
        'nsw-lower_60_second-dispatch': [40.0],
        'nsw-raise_regulation-dispatch': [1.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_load_energy_and_lower_contingency_and_raise_regulation_joint_capacity_con_upper_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-raise_regulation': np.linspace(0, 500, num=101),
        'nsw-lower_60_second': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-lower_60_second-fleet-dispatch': np.zeros(101),
        'nsw-raise_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(30.0)

    u.set_service_region('raise_regulation', 'nsw')
    u.add_regulation_service_to_input('raise_regulation', availability=1.0, ramp_rate=60.0)

    u.set_service_region('lower_60_second', 'nsw')
    u.add_contingency_service_to_input('lower_60_second', availability=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_60_second')
    p.add_regional_market('nsw', 'raise_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [-30.0],
        'nsw-lower_60_second-dispatch': [20.0],
        'nsw-raise_regulation-dispatch': [1.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_generator_energy_and_lower_contingency_and_raise_regulation_joint_capacity_con_lower_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-raise_regulation': np.linspace(0, 500, num=101),
        'nsw-lower_60_second': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-lower_60_second-fleet-dispatch': np.zeros(101),
        'nsw-raise_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(50.0)
    u.add_primary_energy_source(20.0)

    u.set_service_region('raise_regulation', 'nsw')
    u.add_regulation_service_to_output('raise_regulation', availability=1.0, ramp_rate=60.0)

    u.set_service_region('lower_60_second', 'nsw')
    u.add_contingency_service_to_output('lower_60_second', availability=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_60_second')
    p.add_regional_market('nsw', 'raise_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [20.0],
        'nsw-lower_60_second-dispatch': [20.0],
        'nsw-raise_regulation-dispatch': [1.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_generator_energy_and_lower_contingency_and_raise_regulation_joint_capacity_con_plateau():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-raise_regulation': np.linspace(0, 500, num=101),
        'nsw-lower_60_second': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-lower_60_second-fleet-dispatch': np.zeros(101),
        'nsw-raise_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(50.0)
    u.add_primary_energy_source(45.0)

    u.set_service_region('raise_regulation', 'nsw')
    u.add_regulation_service_to_output('raise_regulation', availability=1.0, ramp_rate=60.0)

    u.set_service_region('lower_60_second', 'nsw')
    u.add_contingency_service_to_output('lower_60_second', availability=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_60_second')
    p.add_regional_market('nsw', 'raise_regulation')

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_60_second')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [45.0],
        'nsw-lower_60_second-dispatch': [40.0],
        'nsw-raise_regulation-dispatch': [1.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_generator_energy_and_lower_contingency_and_raise_regulation_joint_capacity_con_upper_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-raise_regulation': np.linspace(0, 500, num=101) * 2,
        'nsw-lower_60_second': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-lower_60_second-fleet-dispatch': np.zeros(101),
        'nsw-raise_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(50.0)
    u.add_primary_energy_source(50.0)

    u.set_service_region('raise_regulation', 'nsw')
    u.add_regulation_service_to_output('raise_regulation', availability=1.0, ramp_rate=60.0)

    u.set_service_region('lower_60_second', 'nsw')
    u.add_contingency_service_to_output('lower_60_second', availability=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_60_second')
    p.add_regional_market('nsw', 'raise_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [49.0],
        'nsw-lower_60_second-dispatch': [40.0],
        'nsw-raise_regulation-dispatch': [1.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_load_energy_and_lower_contingency_and_lower_reg_joint_capacity_con_lower_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-lower_regulation': np.linspace(0, 500, num=101),
        'nsw-lower_60_second': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-lower_60_second-fleet-dispatch': np.zeros(101),
        'nsw-lower_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(0.0)

    u.set_service_region('lower_regulation', 'nsw')
    u.add_regulation_service_to_input('lower_regulation', availability=1.0, ramp_rate=60.0)

    u.set_service_region('lower_60_second', 'nsw')
    u.add_contingency_service_to_input('lower_60_second', availability=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_60_second')
    p.add_regional_market('nsw', 'lower_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [-0.0],
        'nsw-lower_60_second-dispatch': [40.0],
        'nsw-lower_regulation-dispatch': [1.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_load_energy_and_lower_contingency_lower_regulation_joint_capacity_con_plateau():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-lower_regulation': np.linspace(0, 500, num=101),
        'nsw-lower_60_second': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-lower_60_second-fleet-dispatch': np.zeros(101),
        'nsw-lower_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(5.0)

    u.set_service_region('lower_regulation', 'nsw')
    u.add_regulation_service_to_input('lower_regulation', availability=1.0, ramp_rate=60.0)

    u.set_service_region('lower_60_second', 'nsw')
    u.add_contingency_service_to_input('lower_60_second', availability=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_60_second')
    p.add_regional_market('nsw', 'lower_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [-5.0],
        'nsw-lower_60_second-dispatch': [40.0],
        'nsw-lower_regulation-dispatch': [1.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_load_energy_and_lower_contingency_and_lower_regulation_joint_capacity_con_upper_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-lower_regulation': np.linspace(0, 500, num=101) * 10,
        'nsw-lower_60_second': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-lower_60_second-fleet-dispatch': np.zeros(101),
        'nsw-lower_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(30.0)

    u.set_service_region('lower_regulation', 'nsw')
    u.add_regulation_service_to_input('lower_regulation', availability=1.0, ramp_rate=60.0)

    u.set_service_region('lower_60_second', 'nsw')
    u.add_contingency_service_to_input('lower_60_second', availability=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_60_second')
    p.add_regional_market('nsw', 'lower_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [-30.0],
        'nsw-lower_60_second-dispatch': [19.0],
        'nsw-lower_regulation-dispatch': [1.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_generator_energy_and_lower_contingency_and_lower_regulation_joint_capacity_con_lower_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-lower_regulation': np.linspace(0, 500, num=101),
        'nsw-lower_60_second': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-lower_60_second-fleet-dispatch': np.zeros(101),
        'nsw-lower_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(50.0)
    u.add_primary_energy_source(20.0)

    u.set_service_region('lower_regulation', 'nsw')
    u.add_regulation_service_to_output('lower_regulation', availability=1.0, ramp_rate=60.0)

    u.set_service_region('lower_60_second', 'nsw')
    u.add_contingency_service_to_output('lower_60_second', availability=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_60_second')
    p.add_regional_market('nsw', 'lower_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [20.0],
        'nsw-lower_60_second-dispatch': [19.0],
        'nsw-lower_regulation-dispatch': [1.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_generator_energy_and_lower_contingency_and_lower_regulation_joint_capacity_con_plateau():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-lower_regulation': np.linspace(0, 500, num=101),
        'nsw-lower_60_second': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-lower_60_second-fleet-dispatch': np.zeros(101),
        'nsw-lower_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(50.0)
    u.add_primary_energy_source(45.0)

    u.set_service_region('lower_regulation', 'nsw')
    u.add_regulation_service_to_output('lower_regulation', availability=1.0, ramp_rate=60.0)

    u.set_service_region('lower_60_second', 'nsw')
    u.add_contingency_service_to_output('lower_60_second', availability=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_60_second')
    p.add_regional_market('nsw', 'lower_regulation')

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_60_second')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [45.0],
        'nsw-lower_60_second-dispatch': [40.0],
        'nsw-lower_regulation-dispatch': [1.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_generator_energy_and_lower_contingency_and_lower_regulation_joint_capacity_con_upper_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-lower_regulation': np.linspace(0, 500, num=101),
        'nsw-lower_60_second': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-lower_60_second-fleet-dispatch': np.zeros(101),
        'nsw-lower_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(50.0)
    u.add_primary_energy_source(50.0)

    u.set_service_region('lower_regulation', 'nsw')
    u.add_regulation_service_to_output('lower_regulation', availability=1.0, ramp_rate=60.0)

    u.set_service_region('lower_60_second', 'nsw')
    u.add_contingency_service_to_output('lower_60_second', availability=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_60_second')
    p.add_regional_market('nsw', 'lower_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [50.0],
        'nsw-lower_60_second-dispatch': [40.0],
        'nsw-lower_regulation-dispatch': [1.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_load_energy_and_raise_reg_joint_capacity_con_ramping_lower_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-raise_regulation': np.linspace(0, 500, num=101),
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=50.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(30.0)
    u.set_service_region('raise_regulation', 'nsw')
    u.add_regulation_service_to_input('raise_regulation', availability=40.0, ramp_rate=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [-30.0],
        'nsw-raise_regulation-dispatch': [20.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_load_energy_and_raise_reg_joint_capacity_con_ramping_plateau():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-raise_regulation': np.linspace(0, 500, num=101),
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=50.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(45.0)
    u.set_service_region('raise_regulation', 'nsw')
    u.add_regulation_service_to_input('raise_regulation', availability=40.0, ramp_rate=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [-45.0],
        'nsw-raise_regulation-dispatch': [35.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_load_energy_and_raise_reg_joint_capacity_con_ramping_upper_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-raise_regulation': np.linspace(0, 500, num=101),
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=50.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(50.0)
    u.set_service_region('raise_regulation', 'nsw')
    u.add_regulation_service_to_input('raise_regulation', availability=40.0, ramp_rate=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [-50.0],
        'nsw-raise_regulation-dispatch': [40.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_generator_energy_and_raise_reg_joint_capacity_con_ramping_lower_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-raise_regulation': np.linspace(0, 500, num=101),
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(50.0)
    u.add_primary_energy_source(0.0)
    u.set_service_region('raise_regulation', 'nsw')
    u.add_regulation_service_to_output('raise_regulation', availability=40.0, ramp_rate=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [0.0],
        'nsw-raise_regulation-dispatch': [40.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_generator_energy_and_raise_reg_joint_capacity_con_ramping_plateau():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-raise_regulation': np.linspace(0, 500, num=101),
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(50.0)
    u.add_primary_energy_source(5.0)
    u.set_service_region('raise_regulation', 'nsw')
    u.add_regulation_service_to_output('raise_regulation', availability=40.0, ramp_rate=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [5.0],
        'nsw-raise_regulation-dispatch': [35.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_generator_energy_and_raise_reg_joint_capacity_con_ramping_upper_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-raise_regulation': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-raise_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(50.0)
    u.add_primary_energy_source(20.0)
    u.set_service_region('raise_regulation', 'nsw')
    u.add_regulation_service_to_output('raise_regulation', availability=40.0, ramp_rate=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'raise_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [20.0],
        'nsw-raise_regulation-dispatch': [20.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_load_energy_and_lower_reg_joint_capacity_con_ramping_lower_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-lower_regulation': np.linspace(0, 500, num=101),
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-lower_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(0.0)
    u.set_service_region('lower_regulation', 'nsw')
    u.add_regulation_service_to_input('lower_regulation', availability=40.0, ramp_rate=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [0.0],
        'nsw-lower_regulation-dispatch': [40.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_load_energy_and_lower_reg_joint_capacity_con_ramping_plateau():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-lower_regulation': np.linspace(0, 500, num=101),
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-lower_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(10.0)
    u.set_service_region('lower_regulation', 'nsw')
    u.add_regulation_service_to_input('lower_regulation', availability=40.0, ramp_rate=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [-10.0],
        'nsw-lower_regulation-dispatch': [30.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_load_energy_and_lower_reg_joint_capacity_con_ramping_upper_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': -1 * np.linspace(0, 500, num=101),
        'nsw-lower_regulation': np.linspace(0, 500, num=101),
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-lower_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_from_market_energy_flow(50.0)
    u.add_energy_sink(30.0)
    u.set_service_region('lower_regulation', 'nsw')
    u.add_regulation_service_to_input('lower_regulation', availability=40.0, ramp_rate=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [-30.0],
        'nsw-lower_regulation-dispatch': [10.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_generator_energy_and_lower_reg_joint_capacity_con_ramping_lower_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-lower_regulation': np.linspace(0, 500, num=101),
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-lower_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=50.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(50.0)
    u.add_primary_energy_source(20.0)
    u.set_service_region('lower_regulation', 'nsw')
    u.add_regulation_service_to_output('lower_regulation', availability=40.0, ramp_rate=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [20.0],
        'nsw-lower_regulation-dispatch': [10.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_generator_energy_and_lower_reg_joint_capacity_con_ramping_plateau():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-lower_regulation': np.linspace(0, 500, num=101),
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-lower_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=50.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(50.0)
    u.add_primary_energy_source(45.0)
    u.set_service_region('lower_regulation', 'nsw')
    u.add_regulation_service_to_output('lower_regulation', availability=40.0, ramp_rate=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [45.0],
        'nsw-lower_regulation-dispatch': [35.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_generator_energy_and_lower_reg_joint_capacity_con_ramping_upper_slope():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-lower_regulation': np.linspace(0, 500, num=101) * 0.1,
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101),
        'nsw-lower_regulation-fleet-dispatch': np.zeros(101),
    })

    forward_data = pd.DataFrame({
        'interval': [0],
        'nsw-demand': [250]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
                                train_pct=1.0, demand_delta_steps=100)

    u = units.GenericUnit(p, initial_dispatch=50.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(50.0)
    u.add_primary_energy_source(50.0)
    u.set_service_region('lower_regulation', 'nsw')
    u.add_regulation_service_to_output('lower_regulation', availability=40.0, ramp_rate=40.0)

    p.add_regional_market('nsw', 'energy')
    p.add_regional_market('nsw', 'lower_regulation')

    p.optimise()

    dispatch = p.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0],
        'nsw-energy-dispatch': [50.0],
        'nsw-lower_regulation-dispatch': [40.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)
# def test_convergence_across_strongly_linked_energy_markets():
#     """One energy storage unit in each market, 3 dispatch intervals"""
#
#     historical_data = pd.DataFrame({
#         'interval': np.linspace(0, 100, num=101).astype(int),
#         'nsw-demand': np.linspace(0, 500, num=101),
#         'nsw-energy-fleet-dispatch': np.zeros(101),
#         'vic-demand': np.linspace(0, 500, num=101),
#         'vic-energy-fleet-dispatch': np.zeros(101)
#     })
#
#     historical_data['nsw-energy'] = historical_data['nsw-demand'] + historical_data['vic-demand']
#     historical_data['vic-energy'] = historical_data['nsw-demand'] + historical_data['vic-demand']
#
#     forward_data = pd.DataFrame({
#         'interval': [0, 1, 2, 3, 4],
#         'nsw-demand': [100, 400, 400, 400, 400],
#         'vic-demand': [100, 400, 400, 400, 400]})
#
#     p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data,
#                                 train_pct=1.0, demand_delta_steps=100)
#
#     p.add_unit('storage_one', 'nsw')
#     p.add_unit_to_market_flow('storage_one', 50.0)
#     p.add_market_to_unit_flow('storage_one', 50.0)
#     p.add_storage('storage_one', mwh=50.0, initial_mwh=0.0, output_capacity=50.0, input_capacity=50.0,
#                   output_efficiency=1.0, input_efficiency=1.0)
#
#     p.add_regional_market('nsw', 'energy')
#
#     p.add_unit('storage_two', 'vic')
#     p.add_unit_to_market_flow('storage_two', 50.0)
#     p.add_market_to_unit_flow('storage_two', 50.0)
#     p.add_storage('storage_two', mwh=50.0, initial_mwh=0.0, output_capacity=50.0, input_capacity=50.0,
#                   output_efficiency=1.0, input_efficiency=1.0)
#
#     p.add_regional_market('vic', 'energy')
#
#     p.cross_market_optimise()
#
#     dispatch_nsw = p.get_unit_dispatch('storage_one')
#     dispatch_vic = p.get_unit_dispatch('storage_two')
#
#     expect_dispatch = pd.DataFrame({
#         'interval': [0, 1, 2, 3, 4],
#         'net_dispatch': [-50.0, 12.0, 12.0, 13.0, 13.0]
#     })
#
#     assert_frame_equal(expect_dispatch, dispatch_nsw)
#     assert_frame_equal(expect_dispatch, dispatch_vic)
