import pandas as pd
import numpy as np
import math
from itertools import product
from mip import Model, xsum, maximize, INF, BINARY
from causalnex.structure.pytorch import DAGRegressor


class DispatchPlanner:
    def __init__(self, dispatch_interval, forward_data, historical_data=None, train_pct=0.1, demand_delta_steps=10):
        self.dispatch_interval = dispatch_interval
        self.historical_data = historical_data
        self.forward_data = forward_data
        self.planning_horizon = len(self.forward_data.index)
        self.regional_markets = []
        self.price_traces_by_market = {}
        self.units = []
        self.unit_energy_market_mapping = {}
        self.unit_fcas_market_mapping = {}
        self.model = Model(solver_name='CBC', sense='MAX')
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
            price_traces = forecaster.price_forecast_with_demand_sensitivities(forward_data=forward_data, region=region,
                                                                               market=target_column_name,
                                                                               min_delta=-self._get_market_out_flow_capacity(region, service),
                                                                               max_delta=self._get_market_in_flow_capacity(region, service),
                                                                               steps=self.demand_delta_steps)
        else:
            price_traces = forecaster.price_forecast_with_demand_sensitivities(forward_data=forward_data, region=region,
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

        forecaster.train(data=historical_data, train_sample_fraction=0.1, target_col=target_column_name)
        forward_dispatch = forecaster.single_trace_forecast(forward_data=forward_data)

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
    def __init__(self, tabu_child_nodes=['hour', 'dayofweak', 'dayofyear'],
                 tabu_edges=[('demand', 'demand')]):
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

    def train(self, data, train_sample_fraction, target_col):
        self.target_col = target_col
        self.features = [col for col in data.columns
                         if col not in [target_col, 'interval'] and 'fleet-dispatch' not in col]
        tabu_child_nodes = [col for col in self.generic_tabu_edges if col in self.features]
        self.regressor = DAGRegressor(threshold=0.0,
                                      alpha=0.0001,
                                      beta=0.5,
                                      fit_intercept=True,
                                      hidden_layer_units=[5],
                                      standardize=True,
                                      tabu_child_nodes=tabu_child_nodes,
                                      tabu_edges=self._expand_tabu_edges(self.features))
        n_rows = len(data.index)
        sample_size = int(n_rows * train_sample_fraction)
        train = data.sample(sample_size, random_state=1)
        train = train.reset_index(drop=True)
        X, y = train.loc[:, self.features], np.asarray(train[target_col])
        self.regressor.fit(X, y)

    def price_forecast_with_demand_sensitivities(self, forward_data, region, market, min_delta, max_delta, steps):
        prediction = forward_data.loc[:, ['interval']]
        forward_data['old_demand'] = forward_data[region + '-demand'] + forward_data[market + '-fleet-dispatch']
        delta_step_size = max(int((max_delta - min_delta) / steps), 1)
        for delta in range(int(min_delta), int(max_delta) + delta_step_size * 2, delta_step_size):
            forward_data[region + '-demand'] = forward_data['old_demand'] - delta
            X = forward_data.loc[:, self.features]
            Y = self.regressor.predict(X)
            prediction[delta] = Y
        return prediction

    def single_trace_forecast(self, forward_data):
        prediction = forward_data.loc[:, ['interval']]
        X = forward_data.loc[:, self.features]
        Y = self.regressor.predict(X)
        prediction[self.target_col] = Y
        return prediction
