import pandas as pd
import numpy as np
import math
from itertools import product
from mip import Model, xsum, maximize, INF, BINARY, minimize
from causalnex.structure.pytorch import DAGRegressor


class DispatchPlanner:
    def __init__(self, dispatch_interval, forward_data, historical_data=None, train_pct=0.01, demand_delta_steps=10):
        self.dispatch_interval = dispatch_interval
        self.historical_data = historical_data
        self.forward_data = forward_data
        self.planning_horizon = len(self.forward_data.index)
        self.regional_markets = []
        self.price_traces_by_market = {}
        self.units = []
        self.unit_energy_market_mapping = {}
        self.unit_fcas_market_mapping = {}
        self.model = Model(solver_name='GUROBI', sense='MAX')
        self.unit_in_flow_variables = {}
        self.unit_out_flow_variables = {}
        self.units_with_storage = []
        self.unit_storage_input_capacity = {}
        self.unit_storage_output_capacity = {}
        self.unit_storage_input_efficiency = {}
        self.unit_storage_output_efficiency = {}
        self.unit_storage_mwh = {}
        self.unit_storage_initial_mwh = {}
        self.unit_storage_level_variables = {}
        self.unit_output_fcas_variables = {}
        self.unit_input_fcas_variables = {}
        self.unit_initial_mw = {}
        self.market_dispatch_variables = {}
        self.market_net_dispatch_variables = {}
        self.nominal_price_forecast = {}
        self.train_pct = train_pct
        self.demand_delta_steps = demand_delta_steps
        self.expected_regions = ['qld', 'nsw', 'vic', 'sa', 'tas', 'mainland']
        self.expected_service = ['energy',
                                 'raise_5_minute', 'raise_60_second', 'raise_6_second', 'raise_regulation',
                                 'lower_5_minute', 'lower_60_second', 'lower_6_second', 'lower_regulation']
        self.unit_commitment_vars = {}
        self.unit_capacity = {}
        self.unit_min_loading = {}
        self.unit_min_down_time = {}
        self.unit_initial_state = {}
        self.unit_initial_down_time = {}

    def get_model(self):
        return self.model

    def get_planning_horizon(self):
        return int(self.planning_horizon * self.dispatch_interval)

    def get_horizon_in_intervals(self):
        return self.planning_horizon

    def get_time_step(self):
        return self.dispatch_interval

    def add_demand_smoothing_objective_function(self, region, service):
        """Smooths on a daily basis by minimising peaks and maximising troughs. Assumes interval 0 is first interval of
        the day.
        """

        market_name = region + '-' + service
        self.regional_markets.append(market_name)

        if region not in self.market_net_dispatch_variables:
            self.market_net_dispatch_variables[region] = {}

        self.market_net_dispatch_variables[region][service] = {}

        for i in range(0, self.planning_horizon):
            self.market_net_dispatch_variables[region][service][i] = {}

            dispatch_var_name = "net_dispatch_{}_{}".format(market_name, i)
            self.market_net_dispatch_variables[region][service][i] = \
                self.model.add_var(name=dispatch_var_name, lb=-1.0 * INF, ub=INF)

        intervals_per_day = int(60 * 24 / self.get_time_step())

        number_of_days = math.ceil(self.get_horizon_in_intervals() / intervals_per_day)

        demand_series = self.forward_data.set_index('interval')[region + '-demand']

        self.market_peak_demand_cost_vars = {}
        self.market_min_demand_cost_vars = {}

        for day in range(0, number_of_days):
            day_start_interval = day * intervals_per_day
            day_end_interval = min((day + 1) * intervals_per_day, self.forward_data['interval'].max() + 1)
            day_demand_series = demand_series.loc[day_start_interval:day_end_interval]
            day_average_demand = day_demand_series.mean()
            self.market_peak_demand_cost_vars[day] = self.model.add_var(lb=0, ub=INF, obj=-1)
            self.market_min_demand_cost_vars[day] = self.model.add_var(lb=0, ub=INF, obj=-1)

            for i in range(day_start_interval, day_end_interval):
                interval_demand = demand_series.loc[i]
                dispatch = self.market_net_dispatch_variables[region][service][i]
                self.model += (interval_demand - dispatch) - day_average_demand <= self.market_peak_demand_cost_vars[day]
                self.model += day_average_demand - (interval_demand - dispatch) <= self.market_min_demand_cost_vars[day]
                if interval_demand > day_average_demand:
                    self.model += dispatch >= 0
                if interval_demand < day_average_demand:
                    self.model += dispatch <= 0

    def add_demand_smoothing_objective_function_v2(self, region, service):
        """Smooths on a daily basis by minimising peaks and maximising troughs. Assumes interval 0 is first interval of
        the day.
        """

        market_name = region + '-' + service
        self.regional_markets.append(market_name)

        if region not in self.market_net_dispatch_variables:
            self.market_net_dispatch_variables[region] = {}

        self.market_net_dispatch_variables[region][service] = {}

        for i in range(0, self.planning_horizon):
            self.market_net_dispatch_variables[region][service][i] = {}

            dispatch_var_name = "net_dispatch_{}_{}".format(market_name, i)
            self.market_net_dispatch_variables[region][service][i] = \
                self.model.add_var(name=dispatch_var_name, lb=-1.0 * INF, ub=INF)

        intervals_per_day = int(60 * 24 / self.get_time_step())

        number_of_days = math.ceil(self.get_horizon_in_intervals() / intervals_per_day)

        demand_series = self.forward_data.set_index('interval')[region + '-demand']

        self.market_peak_demand_cost_vars = {}
        self.market_min_demand_cost_vars = {}

        for day in range(0, number_of_days):
            day_start_interval = day * intervals_per_day
            day_end_interval = min((day + 1) * intervals_per_day, self.forward_data['interval'].max() + 1)
            self.market_peak_demand_cost_vars[day] = self.model.add_var(lb=0, ub=INF, obj=-1)
            self.market_min_demand_cost_vars[day] = self.model.add_var(lb=0, ub=INF, obj=-1)

            for i in range(day_start_interval, day_end_interval):
                day_demand_series = demand_series.loc[max(i - intervals_per_day / 2, 0):min(i + intervals_per_day / 2 + 1, self.planning_horizon)]
                day_average_demand = day_demand_series.mean()
                interval_demand = demand_series.loc[i]
                dispatch = self.market_net_dispatch_variables[region][service][i]
                self.model += (interval_demand - dispatch) - day_average_demand <= self.market_peak_demand_cost_vars[day]
                self.model += day_average_demand - (interval_demand - dispatch) <= self.market_min_demand_cost_vars[day]
                if interval_demand > day_average_demand:
                    self.model += dispatch >= 0
                if interval_demand < day_average_demand:
                    self.model += dispatch <= 0

    def add_regional_market(self, region, service):
        if self.historical_data is not None:
            self._add_elastic_price_regional_market(region, service)
        else:
            self._add_fixed_price_regional_market(region, service)

    def _add_fixed_price_regional_market(self, region, service):
        market_name = region + '-' + service
        self.regional_markets.append(market_name)

        if region not in self.market_net_dispatch_variables:
            self.market_net_dispatch_variables[region] = {}
        self.market_net_dispatch_variables[region][service] = {}

        forward_prices = self.forward_data.set_index('interval')
        forward_prices = forward_prices[market_name]

        for i in range(0, self.planning_horizon):
            dispatch_var_name = "net_dispatch_{}_{}".format(market_name, i)
            self.market_net_dispatch_variables[region][service][i] = \
                self.model.add_var(name=dispatch_var_name, lb=-1.0 * INF, ub=INF, obj=forward_prices[i])

    def _add_elastic_price_regional_market(self, region, service):
        market_name = region + '-' + service
        self.regional_markets.append(market_name)

        forward_dispatch = self._get_forward_dispatch_trace(region, service, self.forward_data)
        forward_data = pd.merge(self.forward_data, forward_dispatch, on='interval')

        positive_rate, positive_dispatch, negative_rate, negative_dispatch = \
            self._marginal_market_trade(region, service, forward_data)

        if region not in self.market_dispatch_variables:
            self.market_dispatch_variables[region] = {}
        self.market_dispatch_variables[region][service] = {}
        if region not in self.market_net_dispatch_variables:
            self.market_net_dispatch_variables[region] = {}
        self.market_net_dispatch_variables[region][service] = {}

        for i in range(0, self.planning_horizon):
            self.market_dispatch_variables[region][service][i] = {}
            self.market_dispatch_variables[region][service][i]['positive'] = {}
            self.market_dispatch_variables[region][service][i]['negative'] = {}
            self.market_net_dispatch_variables[region][service][i] = {}

            if len(positive_rate) > 0:
                for dispatch, rate in positive_rate[i].items():
                    dispatch_var_name = "dispatch_{}_{}_positive_{}".format(market_name, i, dispatch)
                    self.market_dispatch_variables[region][service][i]['positive'][dispatch] = \
                        self.model.add_var(name=dispatch_var_name, lb=0.0, ub=positive_dispatch[i][dispatch], obj=rate)

            if len(negative_rate) > 0:
                for dispatch, rate in negative_rate[i].items():
                    dispatch_var_name = "dispatch_{}_{}_negative_{}".format(market_name, i, dispatch)
                    self.market_dispatch_variables[region][service][i]['negative'][dispatch] = \
                        self.model.add_var(name=dispatch_var_name, lb=0.0, ub=negative_dispatch[i][dispatch], obj=rate)

            dispatch_var_name = "net_dispatch_{}_{}".format(market_name, i)
            self.market_net_dispatch_variables[region][service][i] = \
                self.model.add_var(name=dispatch_var_name, lb=-1.0 * INF, ub=INF)

            positive_vars = list(self.market_dispatch_variables[region][service][i]['positive'].values())
            negative_vars = list(self.market_dispatch_variables[region][service][i]['negative'].values())
            self.model += xsum([-1 * self.market_net_dispatch_variables[region][service][i]] + positive_vars +
                               [-1 * var for var in negative_vars]) == 0.0

    def _update_price_forecast(self, market, forward_data):
        region = market.split('-')[0]
        service = market.split('-')[1]

        forward_dispatch = self._get_forward_dispatch_trace(region, service, self.forward_data)
        forward_data = pd.merge(self.forward_data, forward_dispatch, on='interval')

        positive_rate, positive_dispatch, negative_rate, negative_dispatch = self._marginal_market_trade(region,
                                                                                                         service,
                                                                                                         forward_data)
        for i in range(0, self.planning_horizon):

            for dispatch, rate in positive_rate[i].items():
                dispatch_var_name = "dispatch_{}_{}_positive_{}".format(market, i, dispatch)
                var = self.model.var_by_name(name=dispatch_var_name)
                var.obj = rate

            if len(negative_rate) > 0:
                for dispatch, rate in negative_rate[i].items():
                    dispatch_var_name = "dispatch_{}_{}_negative_{}".format(market, i, dispatch)
                    var = self.model.var_by_name(name=dispatch_var_name)
                    var.obj = rate

    def _get_revenue_traces(self, region, service, forward_data):
        target_column_name = region + '-' + service

        forecaster = Forecaster()

        cols_to_drop = []
        for region_option, service_option in product(self.expected_regions, self.expected_service):
            col = region_option + '-' + service_option
            if col != target_column_name and col in self.historical_data.columns:
                cols_to_drop.append(col)

        historical_data = self.historical_data.drop(columns=cols_to_drop)
        cols_to_drop = [col for col in cols_to_drop if col in forward_data.columns]
        forward_data = forward_data.drop(columns=cols_to_drop)

        forecaster.train(data=historical_data, train_sample_fraction=self.train_pct, target_col=target_column_name)

        if service == 'energy':
            price_traces = forecaster.price_forecast(forward_data=forward_data, region=region,
                                                     market=target_column_name,
                                                     min_delta=-self._get_market_out_flow_capacity(region, service),
                                                     max_delta=self._get_market_in_flow_capacity(region, service),
                                                     steps=self.demand_delta_steps)
        else:
            price_traces = forecaster.price_forecast(forward_data=forward_data, region=region,
                                                     market=target_column_name,
                                                     min_delta=0,
                                                     max_delta=self._get_market_fcas_capacity(region, service),
                                                     steps=self.demand_delta_steps)

        self.nominal_price_forecast[target_column_name] = price_traces.copy()

        for col in price_traces.columns:
            if col != 'interval':
                price_traces[col] = price_traces[col] * (col + 0.00001)
        return price_traces

    def _get_market_in_flow_capacity(self, region, service):
        capacity = 0.0
        for unit in self.units:
            if unit.service_region_mapping[service] == region and unit.capacity is not None:
                capacity += unit.capacity
        return capacity

    def _get_market_out_flow_capacity(self, region, service):
        capacity = 0.0
        for unit in self.units:
            if unit.service_region_mapping[service] == region:
                if 'market_to_unit' in unit.in_flow_vars:
                    capacity += unit.in_flow_vars['market_to_unit'][0].ub
        return capacity

    def _get_market_fcas_capacity(self, region, service):
        capacity = 0.0
        for unit in self.units:
            if service in unit.service_region_mapping and unit.service_region_mapping[service] == region:
                if service in unit.input_fcas_variables:
                    capacity += unit.input_fcas_variables[service][0].ub
                if service in unit.output_fcas_variables:
                    capacity += unit.output_fcas_variables[service][0].ub
        return capacity

    def _get_forward_dispatch_trace(self, region, service, forward_data):
        target_column_name = region + '-' + service + '-fleet-dispatch'

        forecaster = Forecaster()

        cols_to_drop = []
        for region_option, service_option in product(self.expected_regions, self.expected_service):
            col = region_option + '-' + service_option
            if col in self.historical_data.columns:
                cols_to_drop.append(col)
            col = region_option + '-' + service_option + '-fleet-dispatch'
            if col in self.historical_data.columns and col != target_column_name:
                cols_to_drop.append(col)

        historical_data = self.historical_data.drop(columns=cols_to_drop)
        cols_to_drop = [col for col in cols_to_drop if col in forward_data.columns]
        forward_data = forward_data.drop(columns=cols_to_drop)

        forecaster.train(data=historical_data, train_sample_fraction=self.train_pct, target_col=target_column_name)
        forward_dispatch = forecaster.base_forecast(forward_data=forward_data)

        return forward_dispatch

    def get_nominal_price_forecast(self, region, service):
        return self.nominal_price_forecast[region + '-' + service]

    def _marginal_market_trade(self, region, service, forward_data):
        revenue_trace = self._get_revenue_traces(region, service, forward_data)
        value_columns = [col for col in revenue_trace.columns if col != 'interval']
        stacked = pd.melt(revenue_trace, id_vars=['interval'], value_vars=value_columns,
                          var_name='dispatch', value_name='revenue')

        positive = stacked[stacked['dispatch'] >= 0.0]
        negative = stacked[stacked['dispatch'] <= 0.0].copy()
        negative['dispatch'] = negative['dispatch'] * -1.0

        positive = positive.sort_values('dispatch')
        negative = negative.sort_values('dispatch')

        positive['marginal_revenue'] = positive.groupby('interval', as_index=False)['revenue'].diff()
        negative['marginal_revenue'] = negative.groupby('interval', as_index=False)['revenue'].diff()

        positive['marginal_dispatch'] = positive.groupby('interval', as_index=False)['dispatch'].diff()
        negative['marginal_dispatch'] = negative.groupby('interval', as_index=False)['dispatch'].diff()

        positive = positive[positive['dispatch'] != 0.0]
        negative = negative[negative['dispatch'] != 0.0]

        positive['marginal_rate'] = positive['marginal_revenue'] / positive['marginal_dispatch']
        negative['marginal_rate'] = negative['marginal_revenue'] / negative['marginal_dispatch']

        positive = positive.set_index(['interval', 'dispatch']).loc[:, ['marginal_rate', 'marginal_dispatch']]
        negative = negative.set_index(['interval', 'dispatch']).loc[:, ['marginal_rate', 'marginal_dispatch']]

        positive_rate = positive.groupby(level=0).apply(lambda df: df.xs(df.name).marginal_rate.to_dict()).to_dict()
        positive_dispatch = positive.groupby(level=0).apply(
            lambda df: df.xs(df.name).marginal_dispatch.to_dict()).to_dict()

        negative_rate = negative.groupby(level=0).apply(lambda df: df.xs(df.name).marginal_rate.to_dict()).to_dict()
        negative_dispatch = negative.groupby(level=0).apply(
            lambda df: df.xs(df.name).marginal_dispatch.to_dict()).to_dict()

        return positive_rate, positive_dispatch, negative_rate, negative_dispatch

    def add_unit(self, unit):
        self.units.append(unit)

    def optimise(self):
        for unit in self.units:
            unit.create_constraints_to_balance_unit_energy_flows()
            unit.create_net_output_vars()
        self._create_constraints_to_balance_grid_nodes()
        self.model.optimize()

    def cross_market_optimise(self):
        self.optimise()

        convergence_reached = False

        while not convergence_reached:
            for region_market in self.regional_markets:
                modified_forward_data = self.forward_data
                for region_demand_to_update in self.regional_markets:
                    if 'energy' in region_demand_to_update and region_demand_to_update != region_market:
                        region = region_demand_to_update.split('-')[0]
                        forward_dispatch = self._get_forward_dispatch_trace(region, 'energy', self.forward_data)
                        modified_forward_data = pd.merge(modified_forward_data, forward_dispatch, on='interval')
                        fleet_dispatch_in_region = self.get_market_dispatch(region_demand_to_update)
                        modified_forward_data = pd.merge(modified_forward_data, fleet_dispatch_in_region, on='interval')
                        modified_forward_data[region + '-demand'] = modified_forward_data[region + '-demand'] - \
                                                                    (modified_forward_data['dispatch'] -
                                                                     modified_forward_data[
                                                                         region + '-energy-fleet-dispatch'])
                        modified_forward_data = modified_forward_data.drop(columns='dispatch')
                self._update_price_forecast(region_market, forward_data=modified_forward_data)
            old_dispatch = self.get_dispatch()
            self.model.optimize()
            convergence_reached = self._check_convergence(old_dispatch)

    def _check_convergence(self, previous_dispatch):
        current_dispatch = self.get_dispatch()
        for col in current_dispatch.columns:
            difference = ((current_dispatch[col] - previous_dispatch[col]) / previous_dispatch[col]).abs().max()
            if difference > 0.05:
                return False
        return True

    def _create_constraints_to_balance_grid_nodes(self):
        for market in self.regional_markets:
            region, service = market.split('-')
            for i in range(0, self.planning_horizon):
                net_vars = []
                for unit in self.units:
                    if service in unit.service_region_mapping and unit.service_region_mapping[service] == region:
                        net_vars.append(unit.net_dispatch_vars[service][i])
                self.model += xsum([self.market_net_dispatch_variables[region][service][i]] +
                                   [-1 * var for var in net_vars]) == 0.0

    def _create_constraints_to_balance_unit_nodes(self):
        for unit in self.units:
            for i in range(0, self.planning_horizon):
                in_flow_vars = [var for var_name, var in self.unit_in_flow_variables[unit][i].items()]
                out_flow_vars = [var for var_name, var in self.unit_out_flow_variables[unit][i].items()]
                if unit in self.unit_commitment_vars:
                    min_loading_var = [self.unit_commitment_vars[unit]['state'][i] * self.unit_min_loading[unit] * -1]
                    self.model += xsum(in_flow_vars + [-1 * var for var in out_flow_vars] + min_loading_var) == 0.0
                else:
                    self.model += xsum(in_flow_vars + [-1 * var for var in out_flow_vars]) == 0.0

    def get_storage_energy_flows_and_state_of_charge(self, unit_name):
        if unit_name not in self.units_with_storage:
            raise ValueError('The unit specified does not have a storage component.')

        energy_flows = self.price_traces_by_market[self.regional_markets[0]].loc[:, ['interval']]

    def get_dispatch(self):
        trace = self.forward_data.loc[:, ['interval']]
        for market in self.regional_markets:
            trace[market + '-dispatch'] = \
                trace['interval'].apply(lambda x: self.model.var_by_name(str("net_dispatch_{}_{}".format(market, x))).x,
                                        self.model)
        return trace

    def get_market_dispatch(self, market):
        trace = self.forward_data.loc[:, ['interval']]
        trace['dispatch'] = \
            trace['interval'].apply(lambda x: self.model.var_by_name(str("net_dispatch_{}_{}".format(market, x))).x,
                                    self.model)
        return trace

    def get_fcas_dispatch(self, unit_name):
        trace = self.forward_data.loc[:, ['interval']]
        for service in self.expected_service:
            if service in self.unit_output_fcas_variables[unit_name] and service != 'energy':
                trace[service] = \
                    trace['interval'].apply(
                        lambda x: self.unit_output_fcas_variables[unit_name][service][x].x,)
        return trace

    def get_template_trace(self):
        return self.forward_data.loc[:, ['interval']]



def _create_dispatch_dependent_price_traces(price_forecast, self_dispatch_forecast, capacity_min, capacity_max,
                                            demand_increment):
    """

    Examples
    --------
    >>> price_forecast = pd.DataFrame({
    ...    'interval': [1, 2, 3, 4],
    ...    -30: [100.0, 200.0, 250.0, 500.0],
    ...    -20: [100.0, 200.0, 250.0, 500.0],
    ...    -10: [100.0, 200.0, 250.0, 500.0],
    ...     0:  [100.0, 200.0, 250.0, 500.0],
    ...     10:  [100.0, 200.0, 250.0, 500.0],
    ...     20: [40.0, 80.0, 250.0, 500.0],
    ...     30: [40.0, 80.0, 250.0, 500.0]
    ...     })

    >>> self_dispatch_forecast = pd.DataFrame({
    ...    'interval': [1, 2, 3, 4],
    ...    'dispatch':  [0.0, 10.0, 0.0, -10.0],
    ...    })

    >>> _create_dispatch_dependent_price_traces(price_forecast, self_dispatch_forecast, 0.0, 20.0, 10.0)
       interval   20.0   10.0    0.0
    0         1  100.0  100.0  100.0
    1         2  200.0  200.0  200.0
    2         3  250.0  250.0  250.0
    3         4  500.0  500.0  500.0

    Parameters
    ----------
    sample

    Returns
    -------

    """
    rows = []
    for i in range(0, len(price_forecast['interval'])):
        row = _process_row(price_forecast.iloc[i:i + 1, :], self_dispatch_forecast.iloc[i:i + 1, :], capacity_min,
                           capacity_max, demand_increment)
        rows.append(row)
    return pd.concat(rows)


def _process_row(price_forecast, self_dispatch_forecast, capacity_min, capacity_max, demand_increment):
    """

    Examples
    --------
    >>> price_forecast = pd.DataFrame({
    ...    'interval': [1],
    ...    -20: [100.0],
    ...    -10: [100.0],
    ...     0:  [100.0],
    ...     10:  [100.0],
    ...     20: [40.0]
    ...     })

    >>> self_dispatch_forecast = pd.DataFrame({
    ...    'interval': [1],
    ...    'dispatch':  [10.0],
    ...    })

    >>> _process_row(price_forecast, self_dispatch_forecast, 0.0, 20.0, 10.0)

    Parameters
    ----------
    sample

    Returns
    -------

    """
    dispatch = self_dispatch_forecast['dispatch'].iloc[0]
    demand_deltas = [col for col in price_forecast.columns if col != 'interval']

    # Transform the demand delta columns into absolute dispatch values.
    price_forecast.columns = ['interval'] + [-1.0 * demand_delta + dispatch for demand_delta in demand_deltas]
    dispatch_levels = [col for col in price_forecast.columns if col != 'interval']
    cols_to_keep = ['interval'] + [col for col in dispatch_levels if
                                   capacity_min - demand_increment < col < capacity_max + demand_increment]
    price_forecast = price_forecast.loc[:, cols_to_keep]
    return price_forecast


class Forecaster:
    def __init__(self, tabu_child_nodes=['hour', 'dayofweek', 'dayofyear'],
                 tabu_edges=[('constraint', 'demand'), ('demand', 'demand'),
                             ('constraint', 'constraint'), ('capacity', 'capacity'),
                             ('capacity', 'demand'), ('demand', 'capacity')]):
        self.generic_tabu_child_nodes = tabu_child_nodes
        self.generic_tabu_edges = tabu_edges

    def _expand_tabu_edges(self, data_columns):
        """Prepare the tabu_edges input for the DAGregressor

        Examples
        --------

        >>> f = Forecaster()

        >>> f._expand_tabu_edges(data_columns=['demand-1', 'demand-2', 'constraint-1',
        ...                                    'availability-1', 'availability-2'])

        Parameters
        ----------
        data_columns

        Returns
        -------

        """
        expanded_edges = []
        for generic_edge in self.generic_tabu_edges:
            first_generic_node = generic_edge[0]
            second_generic_node = generic_edge[1]
            specific_first_nodes = [col for col in data_columns if first_generic_node in col]
            specific_second_nodes = [col for col in data_columns if second_generic_node in col]
            specific_edges = product(specific_first_nodes, specific_second_nodes)
            specific_edges = [edge for edge in specific_edges if edge[0] != edge[1]]
            expanded_edges += specific_edges

        return expanded_edges

    def train(self, data, train_sample_fraction, target_col, alpha, beta, layers, bins, sample_size):
        self.target_col = target_col
        self.features = [col for col in data.columns
                         if col not in [target_col, 'interval'] and 'fleet-dispatch' not in col]
        tabu_child_nodes = [col for col in self.generic_tabu_edges if col in self.features]
        self.regressor = DAGRegressor(threshold=0.0,
                                      alpha=alpha,
                                      beta=beta,
                                      fit_intercept=True,
                                      hidden_layer_units=layers,
                                      standardize=True,
                                      tabu_child_nodes=tabu_child_nodes,
                                      tabu_edges=self._expand_tabu_edges(self.features))
        #n_rows = len(data.index)
        #sample_size = int(n_rows * train_sample_fraction)
        #train = data.sample(n=500, replace=True)
        train = self._sample_by_most_recent(data, target_col, bins, sample_size)
        train = train.reset_index(drop=True)
        X, y = train.loc[:, self.features], np.asarray(train[target_col])
        self.X = X
        self.regressor.fit(X, y)

    def _sample(self, data, region):
        data = data.copy()
        data['quantile'] = pd.qcut(data[region + '-demand'], q=10)
        quantile_counts = data.groupby('quantile', as_index=False).agg(
            quantile_count=pd.NamedAgg(column="quantile", aggfunc="count"))
        data = pd.merge(data, quantile_counts, on='quantile')
        data['weight'] = 1 / data['quantile_count']
        data = data.sample(n=500, weights=data['weight'], replace=True)
        return data.drop(columns=['quantile', 'quantile_count', 'weight'])

    def _sample_by_most_recent(self, data, target_col, bins, sample_size):
        data = data.copy()
        data = data[data[target_col] <= 300]
        data = data[data[target_col] >= 0]
        data['bin'] = pd.cut(data[target_col], bins=bins)
        data['sample_order'] = data.groupby('bin').cumcount()
        samples_per_bin = int(sample_size / bins)
        data = data[data['sample_order'] < samples_per_bin]
        return data.drop(columns=['bin'])

    def price_forecast(self, forward_data, region, market, demand_delta):
        prediction = forward_data.loc[:, ['interval']]
        forward_data['old_demand'] = forward_data[region + '-demand'] + forward_data[market + '-fleet-dispatch']
        for delta in demand_delta:
            forward_data[region + '-demand'] = forward_data['old_demand'] + delta
            X = forward_data.loc[:, self.features]
            Y = self.regressor.predict(X)
            prediction[delta] = Y
        return prediction

    def base_forecast(self, forward_data):
        prediction = forward_data.loc[:, ['interval']]
        X = forward_data.loc[:, self.features]
        Y = self.regressor.predict(X)
        prediction[self.target_col] = Y
        return prediction


class ForecastModel:
    def __init__(self, alpha, beta, layers, bins, sample_size, tabu_child_nodes=['hour', 'dayofweek', 'dayofyear'],
                 tabu_edges=[('constraint', 'demand'), ('demand', 'demand'),
                             ('constraint', 'constraint'), ('capacity', 'capacity'),
                             ('capacity', 'demand'), ('demand', 'capacity')]):
        self.generic_tabu_child_nodes = tabu_child_nodes
        self.generic_tabu_edges = tabu_edges
        self.alpha = alpha
        self.beta = beta
        self.layers = layers
        self.bins = bins
        self.sample_size = sample_size
        self.features = None

    def _expand_tabu_edges(self, data_columns):
        """Prepare the tabu_edges input for the DAGregressor

        Examples
        --------

        >>> f = Forecaster()

        >>> f._expand_tabu_edges(data_columns=['demand-1', 'demand-2', 'constraint-1',
        ...                                    'availability-1', 'availability-2'])

        Parameters
        ----------
        data_columns

        Returns
        -------

        """
        expanded_edges = []
        for generic_edge in self.generic_tabu_edges:
            first_generic_node = generic_edge[0]
            second_generic_node = generic_edge[1]
            specific_first_nodes = [col for col in data_columns if first_generic_node in col]
            specific_second_nodes = [col for col in data_columns if second_generic_node in col]
            specific_edges = product(specific_first_nodes, specific_second_nodes)
            specific_edges = [edge for edge in specific_edges if edge[0] != edge[1]]
            expanded_edges += specific_edges

        return expanded_edges

    @staticmethod
    def _sample_by_most_recent(data, target_col, bins, sample_size):
        data = data.copy()
        data = data[data[target_col] <= 300]
        data = data[data[target_col] >= 0]
        data['bin'] = pd.cut(data[target_col], bins=bins)
        data['sample_order'] = data.groupby('bin').cumcount()
        samples_per_bin = int(sample_size / bins)
        data = data[data['sample_order'] < samples_per_bin]
        return data.drop(columns=['bin'])

    @staticmethod
    def _sample_by_most_recent_allow_resample(data, target_col, bins, sample_size):
        data = data.copy()
        data = data[data[target_col] <= 1000]
        data = data[data[target_col] >= -100]
        data['bin'] = pd.cut(data[target_col], bins=bins)
        data['weight'] = (data['interval'] / 288).apply(lambda x: math.floor(x)) + 1
        samples_per_bin = int(sample_size / bins)
        bin_counts = data.groupby('bin', as_index=False)['interval'].count()
        bins_to_use = bin_counts[bin_counts['interval'] >= 30]
        data = data[data['bin'].isin(bins_to_use['bin'])].reset_index(drop=True)
        data = data.groupby('bin', as_index=False, observed=True).sample(n=samples_per_bin, replace=True,
                                                                         weights=data['weight'])
        return data.drop(columns=['bin', 'weight'])

    @staticmethod
    def _sample_across_scenario_variables(data, target_col, bins, sample_size, scenario_variable_columns):
        data = data.copy()
        data = data[data[target_col] <= 1000]
        data = data[data[target_col] >= -100]
        groups = []
        for var in scenario_variable_columns:
            group = 'bin-' + str(var)
            groups.append(group)
            data[group] = pd.cut(data[var], bins=bins)
        data['weight'] = (data['interval'] / 288).apply(lambda x: math.floor(x)) + 1
        bin_counts = data.groupby(groups, as_index=False)['interval'].count()
        bins_to_use = bin_counts[bin_counts['interval'] >= 0].loc[:, groups]
        data = pd.merge(data, bins_to_use, on=groups)
        actual_number_of_bins = len(data.drop_duplicates(subset = groups))
        samples_per_bin = int(sample_size / actual_number_of_bins)
        data = data.groupby(groups, as_index=False, observed=True).sample(n=samples_per_bin, replace=True,
                                                                         weights=data['weight'])
        return data.drop(columns=groups + ['weight'])

    def train(self, data, target_col, scenario_variable_columns):
        self.features = [col for col in data.columns if col not in [target_col, 'interval']]
        tabu_child_nodes = [col for col in self.generic_tabu_child_nodes if col in self.features]
        self.regressor = DAGRegressor(threshold=0.0,
                                      alpha=self.alpha,
                                      beta=self.beta,
                                      fit_intercept=True,
                                      hidden_layer_units=self.layers,
                                      standardize=True,
                                      tabu_child_nodes=tabu_child_nodes,
                                      tabu_edges=self._expand_tabu_edges(self.features))
        train = self._sample_across_scenario_variables(data, target_col, self.bins, self.sample_size, scenario_variable_columns)
        train = train.reset_index(drop=True)
        X, y = train.loc[:, self.features], np.asarray(train[target_col])
        self.regressor.fit(X, y)

    def predict(self, forward_data):
        X = forward_data.loc[:, self.features]
        Y = self.regressor.predict(X)
        return Y


class MultiMarketForecaster:
    def __init__(self, alpha=0.0001, beta=0.6, layers=[5], bins=10, sample_size=5000):
        self.forecast_model_by_market = {}
        self.alpha = alpha
        self.beta = beta
        self.layers = layers
        self.bins = bins
        self.sample_size = sample_size

    def train(self, price_data, regression_features, scenario_variable_columns):
        for market in price_data.columns:
            if market != 'interval':
                self.forecast_model_by_market[market] = ForecastModel(self.alpha, self.beta, self.layers, self.bins,
                                                                      self.sample_size)
                data = pd.merge(price_data.loc[:, ['interval', market]], regression_features, on='interval')
                self.forecast_model_by_market[market].train(data, market, scenario_variable_columns)

    def multi_region_price_forecast(self, forward_data, fleet_deltas):
        feature_columns = [col for col in forward_data.columns if col != 'interval']
        forward_data = generate_forward_sensitivities(forward_data, fleet_deltas)
        #forward_data = forward_data.reset_index(drop=True)
        results = forward_data.copy()
        for market in fleet_deltas.keys():
            features = forward_data.loc[:, feature_columns]
            results[market] = self.forecast_model_by_market[market].predict(features)
        return results


def generate_forward_sensitivities(forward_data, fleet_deltas_by_region):
    """
    Examples
    --------

    >>> data = pd.DataFrame({
    ... 'interval': [0, 1, 2, 3],
    ... 'nsw-demand': [0, 100, 200, 250]})

    >>> deltas_by_region = {'nsw-energy': [0, 20]}

    >>> generate_forward_sensitivities(data, deltas_by_region)
       interval  nsw-demand  scenario  nsw-energy-fleet-delta
    0         0           0         0                       0
    1         0         -20         1                      20
    2         1         100         0                       0
    3         1          80         1                      20
    4         2         200         0                       0
    5         2         180         1                      20
    6         3         250         0                       0
    7         3         230         1                      20


    """
    demand_delta_scenarios = create_dataframe_of_fleet_deltas(fleet_deltas_by_region)
    forward_data = add_forecast_fleet_dispatch_to_regional_demand(forward_data)
    forward_data = pd.merge(forward_data, demand_delta_scenarios, how='cross')
    forward_data = net_off_fleet_deltas_from_regional_demand(forward_data)
    return forward_data


def create_dataframe_of_fleet_deltas(fleet_deltas_by_region):
    """
    Examples
    --------

    >>> deltas = {'nsw-energy': [0, 100], 'vic-energy': [0, 50, 150]}

    >>> create_dataframe_of_fleet_deltas(deltas)
       scenario  nsw-energy-fleet-delta  vic-energy-fleet-delta
    0         0                       0                       0
    1         1                       0                      50
    2         2                       0                     150
    3         3                     100                       0
    4         4                     100                      50
    5         5                     100                     150
    """
    markets = list(fleet_deltas_by_region.keys())
    markets = [market + '-fleet-delta' for market in markets]
    fleet_deltas = list(fleet_deltas_by_region.values())
    rows = product(*fleet_deltas)
    fleet_deltas = pd.DataFrame(data=rows, columns=markets)
    fleet_deltas['scenario'] = fleet_deltas.index
    return fleet_deltas.loc[:, ['scenario'] + markets]


def add_forecast_fleet_dispatch_to_regional_demand(forward_data):
    """
    Examples
    --------

    >>> forecast_data = pd.DataFrame({
    ...  'nsw-demand': [100, 120, 130, 125],
    ...  'nsw-fleet-dispatch': [-10, 20, 40, 20]})

    >>> add_forecast_fleet_dispatch_to_regional_demand(forecast_data)
       nsw-demand  nsw-fleet-dispatch
    0          90                   0
    1         140                  20
    2         170                  40
    3         145                  20

    """
    cols = forward_data.columns
    for col in cols:
        if 'fleet-dispatch' in col:
            region = col.split('-')[0]
            demand_col = region + '-demand'
            forward_data[demand_col] += forward_data[col]
    return forward_data


def net_off_fleet_deltas_from_regional_demand(forward_scenario_data):
    """
    Examples
    --------

    >>> forecast_data = pd.DataFrame({
    ...  'nsw-demand': [100, 120, 130, 125],
    ...  'nsw-energy-fleet-delta': [-10, 20, 40, 20]})

    >>> net_off_fleet_deltas_from_regional_demand(forecast_data)
       nsw-demand  nsw-energy-fleet-delta
    0         110                     -10
    1         100                      20
    2          90                      40
    3         105                      20

    """
    cols = forward_scenario_data.columns
    for col in cols:
        if 'fleet-delta' in col:
            region = col.split('-')[0]
            demand_col = region + '-demand'
            forward_scenario_data[demand_col] -= forward_scenario_data[col]
    return forward_scenario_data


