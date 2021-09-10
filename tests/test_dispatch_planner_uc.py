import pandas as pd
import numpy as np
from pandas._testing import assert_frame_equal
from NEMPRO import planner, units


def test_start_off_with_initial_down_time_of_zero():
    historical_data = pd.DataFrame({
        'interval': np.linspace(0, 100, num=101).astype(int),
        'nsw-energy': np.linspace(0, 500, num=101),
        'nsw-demand': np.linspace(0, 500, num=101),
        'nsw-energy-fleet-dispatch': np.zeros(101)})

    forward_data = pd.DataFrame({
        'interval': [0, 1, 2],
        'nsw-demand': [200, 200, 200]})

    p = planner.DispatchPlanner(dispatch_interval=60, historical_data=historical_data, forward_data=forward_data)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(capacity=100.0)
    u.add_primary_energy_source(capacity=100.0)
    u.add_unit_minimum_operating_level(min_loading=50.0, shutdown_ramp_rate=100.0, start_up_ramp_rate=100.0,
                                       min_up_time=60, min_down_time=120, time_in_initial_state=0)

    p.add_regional_market('nsw', 'energy')

    p.optimise()

    dispatch = u.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0, 1, 2],
        'net_dispatch': [0.0, 0.0, 100.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_start_off_with_initial_down_time_less_than_min_down_time():
    forward_data = pd.DataFrame({
        'interval': [0, 1, 2],
        'nsw-energy': [200, 200, 200]})

    p = planner.DispatchPlanner(dispatch_interval=60, forward_data=forward_data)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(100.0)
    u.add_primary_energy_source(100.0)
    u.add_unit_minimum_operating_level(min_loading=50.0, shutdown_ramp_rate=100.0, start_up_ramp_rate=100.0,
                                       min_up_time=60, min_down_time=120, time_in_initial_state=60)

    p.add_regional_market('nsw', 'energy')

    p.optimise()

    dispatch = u.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0, 1, 2],
        'net_dispatch': [0.0, 100.0, 100.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_start_off_with_initial_down_time_equal_to_min_down_time():
    forward_data = pd.DataFrame({
        'interval': [0, 1, 2],
        'nsw-energy': [200, 200, 200]})

    p = planner.DispatchPlanner(dispatch_interval=60, forward_data=forward_data)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(100.0)
    u.add_primary_energy_source(100.0)
    u.add_unit_minimum_operating_level(min_loading=50.0, shutdown_ramp_rate=100.0, start_up_ramp_rate=100.0,
                                       min_up_time=60, min_down_time=120, time_in_initial_state=120)

    p.add_regional_market('nsw', 'energy')

    p.optimise()

    dispatch = u.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0, 1, 2],
        'net_dispatch': [100.0, 100.0, 100.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_start_on_with_initial_up_time_of_zero():
    forward_data = pd.DataFrame({
        'interval': [0, 1, 2],
        'nsw-energy': [200, 200, 200]})

    p = planner.DispatchPlanner(dispatch_interval=60, forward_data=forward_data)

    u = units.GenericUnit(p, initial_dispatch=50.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(100.0)
    u.add_primary_energy_source(100.0, cost=1000.0)
    u.add_unit_minimum_operating_level(min_loading=50.0, shutdown_ramp_rate=100.0, start_up_ramp_rate=100.0,
                                       min_up_time=120, min_down_time=120, time_in_initial_state=0)

    p.add_regional_market('nsw', 'energy')

    p.optimise()

    dispatch = u.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0, 1, 2],
        'net_dispatch': [50.0, 50.0, 0.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_start_on_with_initial_up_time_less_than_min_up_time():
    forward_data = pd.DataFrame({
        'interval': [0, 1, 2],
        'nsw-energy': [200, 200, 200]})

    p = planner.DispatchPlanner(dispatch_interval=60, forward_data=forward_data)

    u = units.GenericUnit(p, initial_dispatch=50.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(100.0)
    u.add_primary_energy_source(100.0, cost=1000.0)
    u.add_unit_minimum_operating_level(min_loading=50.0, shutdown_ramp_rate=100.0, start_up_ramp_rate=100.0,
                                       min_up_time=120, min_down_time=120, time_in_initial_state=60)

    p.add_regional_market('nsw', 'energy')

    p.optimise()

    dispatch = u.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0, 1, 2],
        'net_dispatch': [50.0, 0.0, 0.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_start_on_with_initial_up_time_equal_to_up_time():
    forward_data = pd.DataFrame({
        'interval': [0, 1, 2],
        'nsw-energy': [200, 200, 200]})

    p = planner.DispatchPlanner(dispatch_interval=60, forward_data=forward_data)

    u = units.GenericUnit(p, initial_dispatch=50.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(100.0)
    u.add_primary_energy_source(100.0, cost=1000.0)
    u.add_unit_minimum_operating_level(min_loading=50.0, shutdown_ramp_rate=100.0, start_up_ramp_rate=100.0,
                                       min_up_time=120, min_down_time=120, time_in_initial_state=120)

    p.add_regional_market('nsw', 'energy')

    p.optimise()

    dispatch = u.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0, 1, 2],
        'net_dispatch': [0.0, 0.0, 0.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_start_on_with_initial_up_time_less_than_min_up_time_check_stays_on():
    forward_data = pd.DataFrame({
        'interval': [0, 1, 2],
        'nsw-energy': [200, 200, 200]})

    p = planner.DispatchPlanner(dispatch_interval=60, forward_data=forward_data)

    u = units.GenericUnit(p, initial_dispatch=50.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(100.0)
    u.add_primary_energy_source(100.0, cost=-500.0)
    u.add_unit_minimum_operating_level(min_loading=50.0, shutdown_ramp_rate=100.0, start_up_ramp_rate=100.0,
                                       min_up_time=120, min_down_time=120, time_in_initial_state=60)

    p.add_regional_market('nsw', 'energy')

    p.optimise()

    dispatch = u.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0, 1, 2],
        'net_dispatch': [100.0, 100.0, 100.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_min_down_time_120_min_constraint():
    forward_data = pd.DataFrame({
        'interval': [0, 1, 2, 3, 4, 5],
        'nsw-energy': [500, 500, 499, 0.0, 500, 500]})

    p = planner.DispatchPlanner(dispatch_interval=60, forward_data=forward_data, demand_delta_steps=10)

    u = units.GenericUnit(p, initial_dispatch=50.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(100.0)
    u.add_primary_energy_source(100.0, cost=400.0)
    u.add_unit_minimum_operating_level(min_loading=50.0, shutdown_ramp_rate=100.0, start_up_ramp_rate=100.0,
                                       min_up_time=120, min_down_time=120, time_in_initial_state=60)

    p.add_regional_market('nsw', 'energy')

    p.optimise()

    dispatch = u.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0, 1, 2, 3, 4, 5],
        'net_dispatch': [100.0, 100.0, 0.0, 0.0, 100.0, 100.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_min_down_time_60_min_constraint():
    forward_data = pd.DataFrame({
        'interval': [0, 1, 2, 3, 4, 5],
        'nsw-energy': [500, 500, 499, 0.0, 500, 500]})

    p = planner.DispatchPlanner(dispatch_interval=60, forward_data=forward_data, demand_delta_steps=10)

    u = units.GenericUnit(p, initial_dispatch=50.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(100.0)
    u.add_primary_energy_source(100.0, cost=400.0)
    u.add_unit_minimum_operating_level(min_loading=50.0, shutdown_ramp_rate=100.0, start_up_ramp_rate=100.0,
                                       min_up_time=120, min_down_time=60, time_in_initial_state=60)

    p.add_regional_market('nsw', 'energy')

    p.optimise()

    dispatch = u.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0, 1, 2, 3, 4, 5],
        'net_dispatch': [100.0, 100.0, 100.0, 0.0, 100.0, 100.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_min_up_time_120_min_constraint():
    forward_data = pd.DataFrame({
        'interval': [0, 1, 2, 3, 4, 5],
        'nsw-energy': [0.0, 0.0, 0.0, 500.0, 1.0, 0.0]})

    p = planner.DispatchPlanner(dispatch_interval=60, forward_data=forward_data, demand_delta_steps=10)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(100.0)
    u.add_primary_energy_source(100.0, cost=50.0)
    u.add_unit_minimum_operating_level(min_loading=50.0, shutdown_ramp_rate=100.0, start_up_ramp_rate=100.0,
                                       min_up_time=120, min_down_time=60, time_in_initial_state=60)

    p.add_regional_market('nsw', 'energy')

    p.optimise()

    dispatch = u.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0, 1, 2, 3, 4, 5],
        'net_dispatch': [0.0, 0.0, 0.0, 100.0, 50.0, 0.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_min_up_time_60_min_constraint():
    forward_data = pd.DataFrame({
        'interval': [0, 1, 2, 3, 4, 5],
        'nsw-energy': [0.0, 0.0, 0.0, 500.0, 1.0, 0.0]})

    p = planner.DispatchPlanner(dispatch_interval=60, forward_data=forward_data, demand_delta_steps=10)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(100.0)
    u.add_primary_energy_source(100.0, cost=50.0)
    u.add_unit_minimum_operating_level(min_loading=50.0, shutdown_ramp_rate=100.0, start_up_ramp_rate=100.0,
                                       min_up_time=60, min_down_time=60, time_in_initial_state=60)

    p.add_regional_market('nsw', 'energy')

    p.optimise()

    dispatch = u.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0, 1, 2, 3, 4, 5],
        'net_dispatch': [0.0, 0.0, 0.0, 100.0, 0.0, 0.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_shutdown_ramp_down_constraint():
    forward_data = pd.DataFrame({
        'interval': [0, 1, 2, 3, 4, 5],
        'nsw-energy': [500, 500, 499, 0.0, 500, 500]})

    p = planner.DispatchPlanner(dispatch_interval=60, forward_data=forward_data, demand_delta_steps=10)

    u = units.GenericUnit(p, initial_dispatch=100.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(100.0)
    u.add_primary_energy_source(100.0, cost=300.0)
    u.add_unit_minimum_operating_level(min_loading=50.0, shutdown_ramp_rate=99.0 / 60, start_up_ramp_rate=100.0 / 60,
                                       min_up_time=60, min_down_time=60, time_in_initial_state=60)

    p.add_regional_market('nsw', 'energy')

    p.optimise()

    dispatch = u.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0, 1, 2, 3, 4, 5],
        'net_dispatch': [100.0, 100.0, 99.0, 0.0, 100.0, 100.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_startup_ramp_up_constraint():
    forward_data = pd.DataFrame({
        'interval': [0, 1, 2, 3, 4, 5],
        'nsw-energy': [0.0, 0.0, 0.0, 500.0, 1.0, 0.0]})

    p = planner.DispatchPlanner(dispatch_interval=60, forward_data=forward_data, demand_delta_steps=10, train_pct=1.0)

    u = units.GenericUnit(p, initial_dispatch=100.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(100.0)
    u.add_primary_energy_source(100.0, cost=50.0)
    u.add_unit_minimum_operating_level(min_loading=50.0, shutdown_ramp_rate=99.0 / 60, start_up_ramp_rate=99.0 / 60,
                                       min_up_time=60, min_down_time=60, time_in_initial_state=60)

    p.add_regional_market('nsw', 'energy')

    p.optimise()

    dispatch = u.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0, 1, 2, 3, 4, 5],
        'net_dispatch': [0.0, 0.0, 0.0, 99.0, 0.0, 0.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_startup_ramp_up_shutdown_ramp_down_constraint():
    forward_data = pd.DataFrame({
        'interval': [0, 1, 2, 3, 4, 5],
        'nsw-energy': [0.0, 0.0, 0.0, 500.0, 1.0, 0.0]})

    p = planner.DispatchPlanner(dispatch_interval=60, forward_data=forward_data, demand_delta_steps=10, train_pct=1.0)

    u = units.GenericUnit(p, initial_dispatch=100.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(100.0)
    u.add_primary_energy_source(100.0, cost=50.0)
    u.add_unit_minimum_operating_level(min_loading=50.0, shutdown_ramp_rate=80.0 / 60, start_up_ramp_rate=99.0 / 60,
                                       min_up_time=60, min_down_time=60, time_in_initial_state=60)

    p.add_regional_market('nsw', 'energy')

    p.optimise()

    dispatch = u.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0, 1, 2, 3, 4, 5],
        'net_dispatch': [0.0, 0.0, 0.0, 99.0, 50.0, 0.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_on_at_start_hot_start_costs_should_turn_on():
    forward_data = pd.DataFrame({
        'interval': [0, 1, 2, 3, 4, 5],
        'nsw-energy': [200.0, 0.0, 0.0, 0.0, 0.0, 200.0]})

    p = planner.DispatchPlanner(dispatch_interval=60, forward_data=forward_data, demand_delta_steps=10, train_pct=1.0)

    u = units.GenericUnit(p, initial_dispatch=100.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(100.0)
    u.add_primary_energy_source(100.0, cost=100.0)

    u.add_unit_minimum_operating_level(min_loading=50.0, shutdown_ramp_rate=100.0, start_up_ramp_rate=100.0,
                                       min_up_time=60, min_down_time=60, time_in_initial_state=60)

    hot_start_cost = 100 * 100 - 1
    cold_start_cost = hot_start_cost * 2

    u.add_startup_costs(hot_start_cost=hot_start_cost, cold_start_cost=cold_start_cost, time_to_go_cold=60 * 10)

    p.add_regional_market('nsw', 'energy')

    p.optimise()

    dispatch = u.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0, 1, 2, 3, 4, 5],
        'net_dispatch': [100.0, 0.0, 0.0, 0.0, 0.0, 100.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_off_at_start_hot_start_costs_should_turn_on():
    forward_data = pd.DataFrame({
        'interval': [0, 1, 2, 3, 4, 5],
        'nsw-energy': [0.0, 0.0, 0.0, 0.0, 0.0, 200.0]})

    p = planner.DispatchPlanner(dispatch_interval=60, forward_data=forward_data, demand_delta_steps=10, train_pct=1.0)

    u = units.GenericUnit(p, initial_dispatch=100.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(100.0)
    u.add_primary_energy_source(100.0, cost=100.0)

    u.add_unit_minimum_operating_level(min_loading=50.0, shutdown_ramp_rate=100.0, start_up_ramp_rate=100.0,
                                       min_up_time=60, min_down_time=60, time_in_initial_state=60)

    hot_start_cost = 100 * 100 - 1
    cold_start_cost = hot_start_cost * 2

    u.add_startup_costs(hot_start_cost=hot_start_cost, cold_start_cost=cold_start_cost, time_to_go_cold=60 * 10)

    p.add_regional_market('nsw', 'energy')

    p.optimise()

    dispatch = u.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0, 1, 2, 3, 4, 5],
        'net_dispatch': [0.0, 0.0, 0.0, 0.0, 0.0, 100.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_off_at_start_cold_start_costs_should_not_turn_on():
    forward_data = pd.DataFrame({
        'interval': [0, 1, 2, 3, 4, 5],
        'nsw-energy': [0.0, 0.0, 0.0, 0.0, 0.0, 200.0]})

    p = planner.DispatchPlanner(dispatch_interval=60, forward_data=forward_data, demand_delta_steps=10, train_pct=1.0)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(100.0)
    u.add_primary_energy_source(100.0, cost=100.0)

    u.add_unit_minimum_operating_level(min_loading=50.0, shutdown_ramp_rate=100.0, start_up_ramp_rate=100.0,
                                       min_up_time=60, min_down_time=60, time_in_initial_state=0)

    hot_start_cost = 100 * 100 - 1
    cold_start_cost = hot_start_cost * 2

    u.add_startup_costs(hot_start_cost=hot_start_cost, cold_start_cost=cold_start_cost, time_to_go_cold=60 * 2)

    p.add_regional_market('nsw', 'energy')

    p.optimise()

    dispatch = u.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0, 1, 2, 3, 4, 5],
        'net_dispatch': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_off_at_start_cold_start_costs_should_turn_on():
    forward_data = pd.DataFrame({
        'interval': [0, 1, 2, 3, 4, 5],
        'nsw-energy': [0.0, 0.0, 0.0, 0.0, 0.0, 200.0]})

    p = planner.DispatchPlanner(dispatch_interval=60, forward_data=forward_data, demand_delta_steps=10, train_pct=1.0)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(100.0)
    u.add_primary_energy_source(100.0, cost=100.0)

    u.add_unit_minimum_operating_level(min_loading=50.0, shutdown_ramp_rate=100.0, start_up_ramp_rate=100.0,
                                       min_up_time=60, min_down_time=60, time_in_initial_state=0)

    hot_start_cost = 100 * 100 - 1
    cold_start_cost = hot_start_cost * 2

    u.add_startup_costs(hot_start_cost=hot_start_cost, cold_start_cost=cold_start_cost, time_to_go_cold=60 * 10)

    p.add_regional_market('nsw', 'energy')

    p.optimise()

    dispatch = u.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0, 1, 2, 3, 4, 5],
        'net_dispatch': [0.0, 0.0, 0.0, 0.0, 0.0, 100.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_initial_down_time_cold_start_costs_should_not_turn_on():
    forward_data = pd.DataFrame({
        'interval': [0, 1, 2, 3, 4, 5],
        'nsw-energy': [200.0, 0.0, 0.0, 0.0, 0.0, 0.0]})

    p = planner.DispatchPlanner(dispatch_interval=60, forward_data=forward_data, demand_delta_steps=10, train_pct=1.0)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(100.0)
    u.add_primary_energy_source(100.0, cost=100.0)

    u.add_unit_minimum_operating_level(min_loading=50.0, shutdown_ramp_rate=100.0, start_up_ramp_rate=100.0,
                                       min_up_time=60, min_down_time=60, time_in_initial_state=60 * 10)

    hot_start_cost = 100 * 100 - 1
    cold_start_cost = hot_start_cost * 2

    u.add_startup_costs(hot_start_cost=hot_start_cost, cold_start_cost=cold_start_cost, time_to_go_cold=60 * 10)

    p.add_regional_market('nsw', 'energy')

    p.optimise()

    dispatch = u.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0, 1, 2, 3, 4, 5],
        'net_dispatch': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)


def test_initial_down_time_cold_start_costs_should_turn_on():
    forward_data = pd.DataFrame({
        'interval': [0, 1, 2, 3, 4, 5],
        'nsw-energy': [200.0, 0.0, 0.0, 0.0, 0.0, 0.0]})

    p = planner.DispatchPlanner(dispatch_interval=60, forward_data=forward_data, demand_delta_steps=10, train_pct=1.0)

    u = units.GenericUnit(p, initial_dispatch=0.0)
    u.set_service_region('energy', 'nsw')
    u.add_to_market_energy_flow(100.0)
    u.add_primary_energy_source(100.0, cost=100.0)

    u.add_unit_minimum_operating_level(min_loading=50.0, shutdown_ramp_rate=100.0, start_up_ramp_rate=100.0,
                                       min_up_time=60, min_down_time=60, time_in_initial_state=60 * 9)

    hot_start_cost = 100 * 100 - 1
    cold_start_cost = hot_start_cost * 2

    u.add_startup_costs(hot_start_cost=hot_start_cost, cold_start_cost=cold_start_cost, time_to_go_cold=60 * 10)

    p.add_regional_market('nsw', 'energy')

    p.optimise()

    dispatch = u.get_dispatch()

    expect_dispatch = pd.DataFrame({
        'interval': [0, 1, 2, 3, 4, 5],
        'net_dispatch': [100.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    })

    assert_frame_equal(expect_dispatch, dispatch)