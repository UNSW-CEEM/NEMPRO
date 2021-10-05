from mip import Model, xsum, maximize, INF, BINARY
import math


class GenericUnit:
    def __init__(self, planner, initial_dispatch=0.0, optimisation_time_step='use_planner_time_step'):
        self.planner = planner
        self.model = planner.get_model()
        planner.add_unit(self)
        self.planning_horizon_minutes = planner.get_planning_horizon()
        self._set_time_step(planner, optimisation_time_step)
        self._set_planning_horizon_in_intervals()

        self.out_flow_vars = {}
        self.in_flow_vars = {}

        self.unit_commitment_vars = {}

        self.output_fcas_variables = {}
        self.input_fcas_variables = {}

        self.net_dispatch_vars = {}

        self.service_region_mapping = {}

        self.capacity = None
        self.initial_mw = initial_dispatch
        self.min_loading = None
        self.min_down_time = None
        self.time_in_initial_state = None
        self.initial_state = None

        self.unit_storage_mwh = None
        self.unit_storage_initial_mwh = None
        self.unit_storage_input_capacity = None
        self.unit_storage_output_capacity = None
        self.unit_storage_input_efficiency = None
        self.unit_storage_output_efficiency = None
        self.unit_storage_level_variables = {}

    def _set_time_step(self, planner, optimisation_time_step):
        if (optimisation_time_step != 'use_planner_time_step' and not
                type(optimisation_time_step) is int):
            raise ValueError("optimisation_time_step should be 'use_planner_time_step' or type int.")

        if optimisation_time_step == 'use_planner_time_step':
            self.time_step = planner.get_time_step()
        else:
            if (optimisation_time_step % planner.get_time_step()) != 0:
                raise ValueError("Planner time step must be multiple of unit time step.")
            if (self.planning_horizon_minutes % optimisation_time_step) != 0:
                raise ValueError("Planning horizon must be multiple of unit time step.")

            self.time_step = optimisation_time_step

    def _set_planning_horizon_in_intervals(self):
        self.planning_horizon_in_intervals = int(self.planning_horizon_minutes / self.time_step)

    def add_to_market_energy_flow(self, capacity):
        self.capacity = capacity
        self.out_flow_vars['unit_to_market'] = {}
        for i in range(0, self.planning_horizon_in_intervals):
            self.out_flow_vars['unit_to_market'][i] = self.model.add_var(ub=capacity)

    def add_from_market_energy_flow(self, capacity):
        self.in_flow_vars['market_to_unit'] = {}
        for i in range(0, self.planning_horizon_in_intervals):
            self.in_flow_vars['market_to_unit'][i] = self.model.add_var(ub=capacity)

    def set_service_region(self, service, region):
        self.service_region_mapping[service] = region

    def add_unit_minimum_operating_level(self, min_loading, shutdown_ramp_rate, start_up_ramp_rate,
                                         min_up_time, min_down_time, time_in_initial_state):
        """Unit commitment constraints are the Tight formulation from Knueven et al. On Mixed Integer Programming
        Formulations for Unit Commitment."""
        if 'market_to_unit' in self.in_flow_vars:
            raise ValueError('Unit commitment constraints not compatible with unit acting as load.')

        startup_max_output = self._mw_per_minute_to_mw_per_interval(start_up_ramp_rate)
        shutdown_max_output = self._mw_per_minute_to_mw_per_interval(shutdown_ramp_rate)
        if startup_max_output < min_loading:
            raise ValueError()
        if shutdown_max_output < min_loading:
            raise ValueError()

        if self.initial_mw < 0:
            raise ValueError("""Unit commitment constraints not compatible with unit acting as load, therefore
                                initial output should not be less than zero.""")

        if 0 < self.initial_mw < min_loading:
            raise ValueError('Unit initial output needs to be either 0 or greater than min_loading.')

        if self.initial_mw < min_loading:
            initial_state = 0
        else:
            initial_state = 1

        self.unit_commitment_vars['state'] = {}
        self.unit_commitment_vars['startup_status'] = {}
        self.unit_commitment_vars['shutdown_status'] = {}
        self.min_loading = min_loading
        self.min_down_time = min_down_time
        self.time_in_initial_state = time_in_initial_state
        self.initial_state = initial_state

        self._create_state_variables()
        self._add_state_variable_constraint(initial_state)
        if initial_state == 1:
            self._add_initial_up_time_constraint(min_up_time, time_in_initial_state)
        elif initial_state == 0:
            self._add_initial_down_time_constraint(min_down_time, time_in_initial_state)
        self._add_min_up_time_constraint(min_up_time)
        self._add_min_down_time_constraint(min_down_time)
        self._update_continuous_production_variable_upper_bound(min_loading)
        self._add_start_up_and_shut_down_ramp_rates(min_loading, startup_max_output, shutdown_max_output)
        self._add_generation_limit_constraint()

    def _mw_per_minute_to_mw_per_interval(self, mw_per_minute):
        return mw_per_minute * self.time_step

    def _create_state_variables(self):
        for i in range(0, self.planning_horizon_in_intervals):
            self.unit_commitment_vars['state'][i] = self.model.add_var(var_type=BINARY)
            self.unit_commitment_vars['startup_status'][i] = self.model.add_var(var_type=BINARY)
            self.unit_commitment_vars['shutdown_status'][i] = self.model.add_var(var_type=BINARY)

    def _add_state_variable_constraint(self, initial_state):
        for i in range(0, self.planning_horizon_in_intervals):
            if i == 0:
                self.model += (self.unit_commitment_vars['state'][i] - initial_state -
                               self.unit_commitment_vars['startup_status'][i] +
                               self.unit_commitment_vars['shutdown_status'][i] == 0)
            else:
                self.model += (self.unit_commitment_vars['state'][i] -
                               self.unit_commitment_vars['state'][i - 1] -
                               self.unit_commitment_vars['startup_status'][i] +
                               self.unit_commitment_vars['shutdown_status'][i] == 0)

    def _add_initial_up_time_constraint(self, min_up_time, initial_up_time):
        min_up_time_in_intervals = self._minutes_to_intervals_round_up(min_up_time)
        initial_up_time_in_intervals = self._minutes_to_intervals_round_up(initial_up_time)
        remaining_up_time = max(0, min_up_time_in_intervals - initial_up_time_in_intervals)
        remaining_up_time = min(remaining_up_time, self.planning_horizon_in_intervals)
        status_vars = []
        for i in range(0, remaining_up_time):
            status_vars.append(self.unit_commitment_vars['state'][i])

        if len(status_vars) > 0:
            self.model += xsum(status_vars) == remaining_up_time

    def _minutes_to_intervals_round_down(self, minutes):
        return math.floor(minutes / self.time_step)

    def _minutes_to_intervals_round_up(self, minutes):
        return math.ceil(minutes / self.time_step)

    def _add_initial_down_time_constraint(self, min_down_time, initial_down_time):
        min_down_time_in_intervals = self._minutes_to_intervals_round_up(min_down_time)
        initial_down_time_in_intervals = self._minutes_to_intervals_round_up(initial_down_time)
        remaining_down_time = max(0, min_down_time_in_intervals - initial_down_time_in_intervals)
        remaining_down_time = min(remaining_down_time, self.planning_horizon_in_intervals)
        status_vars = []
        for i in range(0, remaining_down_time):
            status_vars.append(self.unit_commitment_vars['state'][i])

        self.model += xsum(status_vars) == 0

    def _add_min_up_time_constraint(self, min_up_time):
        min_up_time_in_intervals = self._minutes_to_intervals_round_up(min_up_time)
        for i in range(min_up_time_in_intervals, self.planning_horizon_in_intervals):
            startup_status_vars = []
            for j in range(i - min_up_time_in_intervals + 1, i + 1):
                startup_status_vars.append(self.unit_commitment_vars['startup_status'][j])
            self.model += xsum(startup_status_vars) <= self.unit_commitment_vars['state'][i]

    def _add_min_down_time_constraint(self, min_down_time):
        min_down_time_in_intervals = self._minutes_to_intervals_round_up(min_down_time)
        for i in range(min_down_time_in_intervals, self.planning_horizon_in_intervals):
            shutdown_status_vars = []
            for j in range(i - min_down_time_in_intervals + 1, i + 1):
                shutdown_status_vars.append(self.unit_commitment_vars['shutdown_status'][j])
            self.model += xsum(shutdown_status_vars) <= 1 - self.unit_commitment_vars['state'][i]

    def _update_continuous_production_variable_upper_bound(self, min_loading):
        for i in range(0, self.planning_horizon_in_intervals):
            self.out_flow_vars['unit_to_market'][i].ub = self.capacity - min_loading

    def _add_start_up_and_shut_down_ramp_rates(self, min_loading, startup_max_output, shutdown_max_output):
        continuous_production_capacity = self.capacity - min_loading
        startup_coefficient = self.capacity - startup_max_output
        shutdown_coefficient = max(startup_max_output - shutdown_max_output, 0)
        shutdown_coefficient_2 = self.capacity - shutdown_max_output
        startup_coefficient_2 = max(shutdown_max_output - startup_max_output, 0)

        for i in range(0, self.planning_horizon_in_intervals - 1):
            self.model += (self.out_flow_vars['unit_to_market'][i] -
                           continuous_production_capacity * self.unit_commitment_vars['state'][i] +
                           startup_coefficient * self.unit_commitment_vars['startup_status'][i] +
                           shutdown_coefficient * self.unit_commitment_vars['shutdown_status'][i + 1] <= 0)

            self.model += (self.out_flow_vars['unit_to_market'][i] -
                           continuous_production_capacity * self.unit_commitment_vars['state'][i] +
                           shutdown_coefficient_2 * self.unit_commitment_vars['shutdown_status'][i + 1] +
                           startup_coefficient_2 * self.unit_commitment_vars['startup_status'][i] <= 0)

    def _add_generation_limit_constraint(self):
        for i in range(0, self.planning_horizon_in_intervals):
            self.model += (self.out_flow_vars['unit_to_market'][i] -
                           self.unit_commitment_vars['state'][i] * (self.capacity - self.min_loading)) <= 0.0

    def add_ramp_rates_to_energy_flow_to_market(self, ramp_up_rate, ramp_down_rate):
        max_ramp_up = self._mw_per_minute_to_mw_per_interval(ramp_up_rate)
        max_ramp_down = self._mw_per_minute_to_mw_per_interval(ramp_down_rate)

        min_loading = self.min_loading
        for i in range(0, self.planning_horizon_in_intervals):
            if i == 0:
                self.model += (self.out_flow_vars['unit_to_market'][i] - max(0, self.initial_mw - min_loading) -
                               max_ramp_up <= 0)
                self.model += (max(0, self.initial_mw - min_loading) - self.out_flow_vars['unit_to_market'][i] -
                               max_ramp_down <= 0)
            else:
                self.model += (self.out_flow_vars['unit_to_market'][i] - self.out_flow_vars['unit_to_market'][i - 1] -
                               max_ramp_up <= 0)
                self.model += (self.out_flow_vars['unit_to_market'][i - 1] - self.out_flow_vars['unit_to_market'][i] -
                               max_ramp_down <= 0)

    def add_ramp_rates_to_energy_flow_from_market(self, ramp_up_rate, ramp_down_rate):
        max_ramp_up = self._mw_per_minute_to_mw_per_interval(ramp_up_rate)
        max_ramp_down = self._mw_per_minute_to_mw_per_interval(ramp_down_rate)

        min_loading = self.min_loading
        for i in range(0, self.planning_horizon_in_intervals):
            if i == 0:
                self.model += (self.in_flow_vars['market_to_unit'][i] - max(0, self.initial_mw - min_loading) -
                               max_ramp_up <= 0)
                self.model += (max(0, self.initial_mw - min_loading) - self.in_flow_vars['market_to_unit'][i] -
                               max_ramp_down <= 0)
            else:
                self.model += (self.in_flow_vars['market_to_unit'][i] - self.in_flow_vars['unit_to_market'][i - 1] -
                               max_ramp_up <= 0)
                self.model += (self.in_flow_vars['market_to_unit'][i - 1] - self.in_flow_vars['market_to_unit'][i] -
                               max_ramp_down <= 0)

    def add_startup_costs(self, hot_start_cost, cold_start_cost, time_to_go_cold):
        time_to_go_cold = self._minutes_to_intervals_round_down(time_to_go_cold)
        min_down_time = self._minutes_to_intervals_round_down(self.min_down_time)
        self._add_start_up_costs(hot_start_cost, cold_start_cost, time_to_go_cold, min_down_time)

    def _add_start_up_costs(self, hot_start_cost, cold_start_cost, time_to_go_cold, min_down_time):
        self.unit_commitment_vars['down_time_arc'] = {}
        cost_diff = (hot_start_cost - cold_start_cost)

        if self.initial_state == 0:
            last_shutdown_interval = - self._minutes_to_intervals_round_down(self.time_in_initial_state)
            self.unit_commitment_vars['shutdown_status'][last_shutdown_interval] = \
                self.model.add_var(var_type=BINARY, lb=1)

        for i in range(0, self.planning_horizon_in_intervals):
            self.unit_commitment_vars['down_time_arc'][i] = {}
            for j in range(i - time_to_go_cold, i - min_down_time + 1):
                if j in self.unit_commitment_vars['shutdown_status']:
                    self.unit_commitment_vars['down_time_arc'][i][j] = self.model.add_var(var_type=BINARY,
                                                                                          obj=-cost_diff)

        for i in range(0, self.planning_horizon_in_intervals):
            arc_vars = []
            for j in range(i - time_to_go_cold + 1, i - min_down_time + 1):
                if (i in self.unit_commitment_vars['down_time_arc'] and
                        j in self.unit_commitment_vars['down_time_arc'][i]):
                    arc_vars.append(self.unit_commitment_vars['down_time_arc'][i][j])
            self.model += xsum(arc_vars) - self.unit_commitment_vars['startup_status'][i] <= 0

            arc_vars = []
            for j in range(i + min_down_time, i + time_to_go_cold - 1 + 1):
                if (j in self.unit_commitment_vars['down_time_arc'] and
                        i in self.unit_commitment_vars['down_time_arc'][j]):
                    arc_vars.append(self.unit_commitment_vars['down_time_arc'][j][i])
            self.model += xsum(arc_vars) - self.unit_commitment_vars['shutdown_status'][i] <= 0

            arc_vars = []
            for j in range(i - time_to_go_cold + 1, i - min_down_time + 1):
                if (i in self.unit_commitment_vars['down_time_arc'] and
                        j in self.unit_commitment_vars['down_time_arc'][i]):
                    arc_vars.append(self.unit_commitment_vars['down_time_arc'][i][j])

            self.unit_commitment_vars['startup_status'][i].obj = -cold_start_cost


    def add_capacity_constraints_on_output(self, service, max_available, enablement_min,
                                           low_breakpoint, high_breakpoint, enablement_max):

        if "unit_to_market" in self.out_flow_vars:
            upper_slope_coefficient = (enablement_max - high_breakpoint) / max_available
            lower_slope_coefficient = (low_breakpoint - enablement_min) / max_available

            for i in range(0, self.planning_horizon_in_intervals):
                energy_dispatch_target = self.out_flow_vars["unit_to_market"][i]
                fcas_contingency_target = self.output_fcas_variables[service][i]

                self.model += energy_dispatch_target + upper_slope_coefficient * fcas_contingency_target \
                              <= enablement_max

                self.model += energy_dispatch_target - lower_slope_coefficient * fcas_contingency_target \
                              >= enablement_min

    def add_capacity_constraints_on_input(self, service, max_available, enablement_min,
                                          low_breakpoint, high_breakpoint, enablement_max):

        if "market_to_unit" in self.in_flow_vars:
            upper_slope_coefficient = (enablement_max - high_breakpoint) / max_available
            lower_slope_coefficient = (low_breakpoint - enablement_min) / max_available

            for i in range(0, self.planning_horizon_in_intervals):
                energy_dispatch_target = self.in_flow_vars["market_to_unit"][i]
                fcas_contingency_target = self.input_fcas_variables[service][i]

                self.model += energy_dispatch_target + upper_slope_coefficient * fcas_contingency_target \
                              <= enablement_max

                self.model += energy_dispatch_target - lower_slope_coefficient * fcas_contingency_target \
                              >= enablement_min

    def add_storage(self, mwh, initial_mwh, output_capacity, output_efficiency,
                    input_capacity, input_efficiency):

        self.unit_storage_mwh = mwh
        self.unit_storage_initial_mwh = initial_mwh
        self.unit_storage_input_capacity = input_capacity
        self.unit_storage_output_capacity = output_capacity
        self.unit_storage_input_efficiency = input_efficiency
        self.unit_storage_output_efficiency = output_efficiency

        self.out_flow_vars['unit_to_storage'] = {}
        self.in_flow_vars['storage_to_unit'] = {}

        for i in range(0, self.planning_horizon_in_intervals):
            self.out_flow_vars['unit_to_storage'][i] = self.model.add_var(ub=input_capacity)
            self.in_flow_vars['storage_to_unit'][i] = self.model.add_var(ub=output_capacity)
            self.unit_storage_level_variables[i] = self.model.add_var(ub=mwh)
            input_to_storage = self.out_flow_vars['unit_to_storage'][i]
            output_from_storage = self.in_flow_vars['storage_to_unit'][i]
            storage_level = self.unit_storage_level_variables[i]
            hours_per_interval = self.time_step / 60

            if i == 0:
                self.model += initial_mwh - (output_from_storage / output_efficiency) * hours_per_interval + \
                              (input_to_storage * input_efficiency) * hours_per_interval == storage_level
            else:
                previous_storage_level = self.unit_storage_level_variables[i - 1]
                self.model += previous_storage_level - (output_from_storage / output_efficiency) * hours_per_interval + \
                              (input_to_storage * input_efficiency) * hours_per_interval == storage_level

    def add_primary_energy_source(self, capacity, cost=0.0):
        self.in_flow_vars['generator_to_unit'] = {}
        for i in range(0, self.planning_horizon_in_intervals):
            self.in_flow_vars['generator_to_unit'][i] = self.model.add_var(ub=capacity, obj=-1 * cost)

    def add_energy_sink(self, capacity, cost=0.0):
        self.out_flow_vars['unit_to_load'] = {}
        for i in range(0, self.planning_horizon_in_intervals):
            self.out_flow_vars['unit_to_load'][i] = self.model.add_var(ub=capacity, obj=-1 * cost)

    def create_constraints_to_balance_unit_energy_flows(self):
        for i in range(0, self.planning_horizon_in_intervals):
            in_flow_vars = [var_dict[i] for var_types, var_dict in self.in_flow_vars.items()]
            out_flow_vars = [var_dict[i] for var_types, var_dict in self.out_flow_vars.items()]
            if 'state' in self.unit_commitment_vars:
                min_loading_var = [self.unit_commitment_vars['state'][i] * self.min_loading * -1]
                self.model += xsum(in_flow_vars + [-1 * var for var in out_flow_vars] + min_loading_var) == 0.0
            else:
                self.model += xsum(in_flow_vars + [-1 * var for var in out_flow_vars]) == 0.0

    def create_net_output_vars(self):
        for service in self.service_region_mapping.keys():
            self.net_dispatch_vars[service] = {}
            for i in range(0, self.planner.get_horizon_in_intervals()):
                self.net_dispatch_vars[service][i] = self.model.add_var(lb=-INF,  ub=INF)
                lhs = [self.net_dispatch_vars[service][i]]
                rate = self.time_step / self.planner.get_time_step()
                if (i+1) % rate == 0:
                    if service == 'energy':
                        unit_index = int(i/rate)
                        if 'state' in self.unit_commitment_vars:
                            lhs.append(self.unit_commitment_vars['state'][unit_index] * self.min_loading * -1)
                        if 'unit_to_market' in self.out_flow_vars:
                            lhs.append(self.out_flow_vars['unit_to_market'][unit_index] * -1)
                        if 'market_to_unit' in self.in_flow_vars:
                            lhs.append(self.in_flow_vars['market_to_unit'][unit_index])
                        self.model += xsum(lhs) == 0.0
                    else:
                        if service in self.output_fcas_variables:
                            lhs.append(self.output_fcas_variables[service][unit_index] * -1)
                        if service in self.input_fcas_variables:
                            lhs.append(self.input_fcas_variables[service][unit_index] * -1)
                        if len(lhs) > 1:
                            self.model += xsum(lhs) == 0.0
                else:
                    next_unit_interval = math.ceil((i + 1) / rate) - 1
                    next_interval_weight = ((i + 1) % rate) / rate
                    previous_unit_interval = next_unit_interval - 1
                    previous_interval_weight = 1 - next_interval_weight
                    if service == 'energy':
                        if i < rate:
                            lhs.append(self.initial_mw * -1 * previous_interval_weight)
                            if 'state' in self.unit_commitment_vars:
                                lhs.append(self.unit_commitment_vars['state'][next_unit_interval] * self.min_loading * -1
                                           * next_interval_weight)
                            if 'unit_to_market' in self.out_flow_vars:
                                lhs.append(self.out_flow_vars['unit_to_market'][next_unit_interval] * -1 *
                                           next_interval_weight)
                            if 'market_to_unit' in self.in_flow_vars:
                                lhs.append(self.in_flow_vars['market_to_unit'][next_unit_interval] * next_interval_weight)
                            self.model += xsum(lhs) == 0.0
                        else:
                            if 'state' in self.unit_commitment_vars:
                                lhs.append(self.unit_commitment_vars['state'][previous_unit_interval] * self.min_loading * -1
                                           * previous_interval_weight)
                                lhs.append(self.unit_commitment_vars['state'][next_unit_interval] * self.min_loading * -1
                                           * next_interval_weight)
                            if 'unit_to_market' in self.out_flow_vars:
                                lhs.append(self.out_flow_vars['unit_to_market'][previous_unit_interval] * -1 *
                                           previous_interval_weight)
                                lhs.append(self.out_flow_vars['unit_to_market'][next_unit_interval] * -1 *
                                           next_interval_weight)
                            if 'market_to_unit' in self.in_flow_vars:
                                lhs.append(self.in_flow_vars['market_to_unit'][previous_unit_interval] * previous_interval_weight)
                                lhs.append(self.in_flow_vars['market_to_unit'][next_unit_interval] * next_interval_weight)
                            self.model += xsum(lhs) == 0.0
                    else:
                        if service in self.output_fcas_variables:
                            lhs.append(self.output_fcas_variables[service][next_unit_interval] * -1)
                        if service in self.input_fcas_variables:
                            lhs.append(self.input_fcas_variables[service][next_unit_interval] * -1)
                        if len(lhs) > 1:
                            self.model += xsum(lhs) == 0.0

    def get_dispatch(self, service='energy'):
        energy_flows = self.planner.get_template_trace()
        energy_flows['net_dispatch'] = energy_flows['interval'].apply(lambda x: self.net_dispatch_vars[service][x].x)
        return energy_flows

    def get_unit_energy_flows(self):
        energy_flows = self.planner.get_template_trace()

        if 'unit_to_market' in self.out_flow_vars[0]:
            energy_flows['unit_to_market'] = \
                energy_flows['interval'].apply(lambda x: self.out_flow_vars['unit_to_market'][x].x)

        if 'market_to_unit' in self.in_flow_vars[0]:
            energy_flows['market_to_unit'] = \
                energy_flows['interval'].apply(lambda x: self.in_flow_vars['market_to_unit'][x].x)

        if 'generator_to_unit' in self.in_flow_vars[0]:
            energy_flows['generator_to_unit'] = \
                energy_flows['interval'].apply(lambda x: self.in_flow_vars['generator_to_unit'][x].x)

        if 'state' in self.unit_commitment_vars:
            energy_flows['state'] = \
                energy_flows['interval'].apply(lambda x: self.unit_commitment_vars['state'][x].x)

        if 'startup_status' in self.unit_commitment_vars:
            energy_flows['startup_status'] = \
                energy_flows['interval'].apply(lambda x: self.unit_commitment_vars['startup_status'][x].x)

        if 'shutdown_status' in self.unit_commitment_vars:
            energy_flows['shutdown_status'] = \
                energy_flows['interval'].apply(lambda x: self.unit_commitment_vars['shutdown_status'][x].x)

        energy_flows['net_dispatch'] = 0.0

        if 'unit_to_market' in energy_flows.columns:
            energy_flows['net_dispatch'] += energy_flows['unit_to_market']

        if 'market_to_unit' in energy_flows.columns:
            energy_flows['net_dispatch'] -= energy_flows['market_to_unit']

        if 'state' in energy_flows.columns:
            energy_flows['net_dispatch'] += energy_flows['state'] * self.min_loading

        return energy_flows


