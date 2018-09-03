import numpy as np
from numpy import random
import psyneulink as pnl
from collections import defaultdict
from scipy import io
import pandas


# TODO: adjust if necessary
STROOP_NUM_FEATURES = 2
STROOP_HIDDEN_LAYER_SIZE = 2
STROOP_LEARNING_RATE = 0.02
STROOP_HIDDEN_BIAS = 4  # PNL bias is subtracted, so this mean a bias of -4
STROOP_HIDDEN_GAIN = 1
STROOP_INTEGRATION_RATE = 0.1
STROOP_INTEGRATOR_MODE = True
STROOP_NOISE_STD = 0.005

STROOP_ACCUMULATOR_RATE = 0.2
STROOP_ACCUMULATOR_NOISE_STD = 0.01
STROOP_ACCUMULATOR_THRESHOLD = 1.0

STROOP_WEIGHT_INIT_SCALE = 2e-2

STROOP_FIRST_PATHWAY_NAME = 'color'
STROOP_SECOND_PATHWAY_NAME = 'word'
STROOP_NAME = 'Stroop'

# Loaded weight parameters
FIRST_PATHWAY_HIDDEN_KEY = 'WeightsFirstPathwayHidden'
FIRST_PATHWAY_TASK_HIDDEN_KEY = 'WeightsFirstPathwayTask'

SECOND_PATHWAY_HIDDEN_KEY = 'WeightsSecondPathwayHidden'
SECOND_PATHWAY_TASK_HIDDEN_KEY = 'WeightsSecondPathwayTask'

FIRST_PATHWAY_OUTPUT_KEY = 'WeightsFirstPathwayOutput'
SECOND_PATHWAY_OUTPUT_KEY = 'WeightsSecondPathwayOutput'

FIRST_ACCUMULATOR_DIFFERENCING_KEY = 'FirstAccumulatorDifferencing'
SECOND_ACCUMULATOR_DIFFERENCING_KEY = 'SecondAccumulatorDifferencing'

"""
Shape to hidden unites are described as "initial connection strengths from the input to the intermediate units
that allowed it to generate a useful representation at the level of the intermediate units" -- 
so maybe a bit weaker than color?

Words are 2.6, colors 2.2, so let's try 1.8?
"""

STROOP_WEIGHT_DICT = {
    FIRST_PATHWAY_HIDDEN_KEY: np.matrix([[2.2, -2.2],
                                         [-2.2, 2.2]]),
    SECOND_PATHWAY_HIDDEN_KEY: np.matrix([[2.6, -2.6],
                                          [-2.6, 2.6]]),
    FIRST_PATHWAY_TASK_HIDDEN_KEY: np.matrix([[4.0, 4.0]]),
    SECOND_PATHWAY_TASK_HIDDEN_KEY: np.matrix([[4.0, 4.0]]),
    FIRST_PATHWAY_OUTPUT_KEY: np.matrix([[1.3, -1.3],
                                         [-1.3, 1.3]]),
    SECOND_PATHWAY_OUTPUT_KEY: np.matrix([[2.5, -2.5],
                                          [-2.5, 2.5]]),
    FIRST_ACCUMULATOR_DIFFERENCING_KEY: np.matrix([[1.0], [-1.0]]),
    SECOND_ACCUMULATOR_DIFFERENCING_KEY: np.matrix([[-1.0], [1.0]])
}

# Training-related constants
STROOP_TRAIN_TRIALS_PER_STIMULUS = 100
STROOP_TEST_TRIALS_PER_STIMULUS = 100
CONTROL_CONDITION = 'control'
CONGRUENT_CONDITION = 'congruent'
CONFLICT_CONDITION = 'conflict'


class StroopModel:
    def __init__(self, num_features=STROOP_NUM_FEATURES, weight_file=None, weight_dict=STROOP_WEIGHT_DICT,
                 log_values=True, *,
                 hidden_layer_size=STROOP_HIDDEN_LAYER_SIZE, hidden_bias=STROOP_HIDDEN_BIAS,
                 hidden_gain=STROOP_HIDDEN_GAIN, integration_rate=STROOP_INTEGRATION_RATE,
                 integrator_mode=STROOP_INTEGRATOR_MODE, noise_std=STROOP_NOISE_STD,
                 learning_rate=STROOP_LEARNING_RATE,
                 accumulator_rate=STROOP_ACCUMULATOR_RATE, accumulator_noise_std=STROOP_ACCUMULATOR_NOISE_STD,
                 accumulator_threshold=STROOP_ACCUMULATOR_THRESHOLD, weight_init_scale=STROOP_WEIGHT_INIT_SCALE,
                 first_pathway_name=STROOP_FIRST_PATHWAY_NAME, second_pathway_name=STROOP_SECOND_PATHWAY_NAME,
                 name=STROOP_NAME):

        self.num_features = num_features
        self.log_values = log_values

        if weight_file is not None:
            self.loaded_weights = io.loadmat(weight_file)
            # Transpose values saved from Matlab, since they use the opposite order
            for key in self.loaded_weights.keys():
                if isinstance(self.loaded_weights[key], np.ndarray):
                    self.loaded_weights[key] = self.loaded_weights[key].T

        elif weight_dict is not None:
            self.loaded_weights = weight_dict

        else:
            self.loaded_weights = None

        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate
        self.integration_rate = integration_rate
        self.integrator_mode = integrator_mode
        self.noise_std = noise_std
        self.hidden_bias = hidden_bias
        self.hidden_gain = hidden_gain

        self.accumulator_rate = accumulator_rate
        self.accumulator_noise_std = accumulator_noise_std
        self.accumulator_threshold = accumulator_threshold

        self.weight_init_scale = weight_init_scale
        self.first_pathway_name = first_pathway_name
        self.second_pathway_name = second_pathway_name
        self.name = name

        self._generate_layers()
        self._add_layer_logging()
        self._name_layers()
        self._generate_processes()
        self._name_processes()
        self._generate_system()

    def _generate_noise_function(self):
        """
        Generate the noise function with the supplied `noise_std`, split to a member since this tends to recurr.
        :return: A PsyNeuLink noise function with a normal noise distribution
        """
        return pnl.NormalDist(standard_dev=self.noise_std).function

    def _generate_layers(self):
        """
        Generate the layers for this model. The hidden layers use an integrator mode, rate, and noise function.
        :return: None, saves the layers into a whole bunch of members
        """
        # Inputs
        self._first_pathway_input_layer = pnl.TransferMechanism(size=self.num_features,
                                                                name=f'{self.first_pathway_name}_input')
        self._second_pathway_input_layer = pnl.TransferMechanism(size=self.num_features,
                                                                 name=f'{self.second_pathway_name}_input')

        # Task units
        self._first_pathway_task_layer = pnl.TransferMechanism(size=1, name=f'{self.first_pathway_name}_task')
        self._second_pathway_task_layer = pnl.TransferMechanism(size=1, name=f'{self.second_pathway_name}_task')

        # Hidden layers
        self._first_pathway_hidden_layer = pnl.TransferMechanism(
            size=self.hidden_layer_size, name=f'{self.first_pathway_name}_hidden',
            function=pnl.Logistic(gain=self.hidden_gain, bias=self.hidden_bias),
            integrator_mode=self.integrator_mode, integration_rate=self.integration_rate,
            noise=self._generate_noise_function())
        self._second_pathway_hidden_layer = pnl.TransferMechanism(
            size=self.hidden_layer_size, name=f'{self.second_pathway_name}_hidden',
            function=pnl.Logistic(gain=self.hidden_gain, bias=self.hidden_bias),
            integrator_mode=self.integrator_mode, integration_rate=self.integration_rate,
            noise=self._generate_noise_function())

        # Output layers
        self.output_layer = pnl.TransferMechanism(size=self.num_features,
                                                  name='output',
                                                  function=pnl.Logistic,
                                                  integrator_mode=self.integrator_mode,
                                                  integration_rate=self.integration_rate,
                                                  noise=self._generate_noise_function())

        self.first_accumulator = pnl.IntegratorMechanism(
            function=pnl.SimpleIntegrator(noise=pnl.NormalDist(standard_dev=self.accumulator_noise_std).function,
                                          rate=self.accumulator_rate),
            name='first_response_accumulator')

        self.second_accumulator = pnl.IntegratorMechanism(
            function=pnl.SimpleIntegrator(noise=pnl.NormalDist(standard_dev=self.accumulator_noise_std).function,
                                          rate=self.accumulator_rate),
            name='second_response_accumulator')

    def _add_layer_logging(self):
        if self.log_values:
            self.log_layers = [self._first_pathway_hidden_layer, self._second_pathway_hidden_layer,
                               self.output_layer, self.first_accumulator, self.second_accumulator]

            for layer in self.log_layers:
                layer.set_log_conditions('value')

    def _name_layers(self):
        self.color_input_layer = self._first_pathway_input_layer
        self.word_input_layer = self._second_pathway_input_layer

        self.color_task_layer = self._first_pathway_task_layer
        self.word_task_layer = self._second_pathway_task_layer

        self.color_hidden_layer = self._first_pathway_hidden_layer
        self.word_hidden_layer = self._second_pathway_hidden_layer

    def _should_load(self, key):
        """
        A utility to help figure out when should use pre-provided weights, instead of generating random ones.
        We use loaded weights if any loaded weights exist and the particular key we're seeking is saved there.
        :param key: The key to retrieve loaded weights for
        :return: A boolean indicator for whether or not weights need to be loaded.
        """
        return self.loaded_weights is not None and key in self.loaded_weights

    def _generate_process(self, weight_key, in_size, out_size, in_layer, out_layer, name, learning_rate=0.0):
        """
        Generate a simple PNL process between two layers.
        :param weight_key: The key to use to load weights for this network, if pre-loaded weights exist
        :param in_size: The size of the input to this process
        :param out_size: The size of the output from this process
        :param in_layer: The layer (`pnl.TransferMechanism`) object that is the input to this process
        :param out_layer: The layer (`pnl.TransferMechanism`) object that is the output to this process
        :param name: A name to give this layer
        :param learning_rate: A learning rate for this process. If none given, remains zero.
        :return: The generated process specified by the input parameters
        """
        if self._should_load(weight_key):
            weights = self.loaded_weights[weight_key]
        else:
            weights = pnl.random_matrix(in_size, out_size, 2, -1) * self.weight_init_scale

        learning = learning_rate != 0
        process = pnl.Process(pathway=[in_layer, weights, out_layer],
                              name=name, learning=learning and pnl.LEARNING or None)
        self.processes.append(process)

        if learning:
            learning_mechanism = process.pathway[1].learning_mechanism
            learning_mechanism.learning_rate = learning_rate
            self.learning_mechanisms_to_learning_rates[learning_mechanism] = learning_rate

        return process

    def _generate_processes(self):
        """
        Generate all processes.
        :return: None; processes generated and saved to members
        """
        self.processes = []
        self.learning_mechanisms_to_learning_rates = {}

        self._first_pathway_hidden_process = self._generate_process(
            FIRST_PATHWAY_HIDDEN_KEY, self.num_features, self.hidden_layer_size,
            self._first_pathway_input_layer, self._first_pathway_hidden_layer,
            f'{self.first_pathway_name}-hidden-process', self.learning_rate)

        self._first_pathway_task_process = self._generate_process(
            FIRST_PATHWAY_TASK_HIDDEN_KEY, 1, self.hidden_layer_size,
            self._first_pathway_task_layer, self._first_pathway_hidden_layer,
            f'task-{self.first_pathway_name}-hidden-process')

        self._first_pathway_output_process = self._generate_process(
            FIRST_PATHWAY_OUTPUT_KEY, self.hidden_layer_size, self.num_features,
            self._first_pathway_hidden_layer, self.output_layer, f'{self.first_pathway_name}-output-process',
            self.learning_rate)

        self._second_pathway_hidden_process = self._generate_process(
            SECOND_PATHWAY_HIDDEN_KEY, self.num_features, self.hidden_layer_size,
            self._second_pathway_input_layer, self._second_pathway_hidden_layer,
            f'{self.second_pathway_name}-hidden-process', self.learning_rate)

        self._second_pathway_task_process = self._generate_process(
            SECOND_PATHWAY_TASK_HIDDEN_KEY, 1, self.hidden_layer_size,
            self._second_pathway_task_layer, self._second_pathway_hidden_layer,
            f'task-{self.second_pathway_name}-hidden-process')

        self._second_pathway_output_process = self._generate_process(
            SECOND_PATHWAY_OUTPUT_KEY, self.hidden_layer_size, self.num_features,
            self._second_pathway_hidden_layer, self.output_layer, f'{self.second_pathway_name}-output-process',
            self.learning_rate)

        self.first_accumulator_process = self._generate_process(FIRST_ACCUMULATOR_DIFFERENCING_KEY,
                                                                self.num_features, self.num_features,
                                                                self.output_layer, self.first_accumulator,
                                                                'first-accumulator-process')

        self.second_accumulator_process = self._generate_process(SECOND_ACCUMULATOR_DIFFERENCING_KEY,
                                                                 self.num_features, self.num_features,
                                                                 self.output_layer, self.second_accumulator,
                                                                 'second-accumulator-process')

        self.integrating_mechanisms = [self._first_pathway_hidden_layer,
                                       self._second_pathway_hidden_layer,
                                       self.output_layer]

    def _name_processes(self):
        self.color_hidden_process = self._first_pathway_hidden_process
        self.word_hidden_process = self._second_pathway_hidden_process

        self.color_task_process = self._first_pathway_task_process
        self.word_task_process = self._second_pathway_task_process

        self.color_output_process = self._first_pathway_output_process
        self.word_output_process = self._second_pathway_output_process

    def _generate_system(self):
        """
        Generate the overall PNL system for this model. Given that all of the heavy lifting is done above,
        this doesn't really do anything.
        :return: None; PsyNeuLink system generated.
        """
        self.system = pnl.System(
            name=self.name,
            processes=self.processes
        )

    def _switch_trial_type(self):
        if isinstance(self.system.termination_processing[pnl.TimeScale.TRIAL], pnl.AllHaveRun):
            self._switch_to_integration_trial()
        else:
            self._switch_trial_settings()

    def _switch_to_integration_trial(self):
        def pass_threshold(first_mechanism, second_mechanism, threshold):
            if np.any(first_mechanism.output_states[0].value >= threshold) or \
                    np.any(second_mechanism.output_states[0].value >= threshold):
                return True

            return False

        self._switch_trial_settings(True, self._generate_noise_function(),
                                    pnl.While(pass_threshold, self.first_accumulator,
                                              self.second_accumulator, self.accumulator_threshold), True)

    def _switch_trial_settings(self, integrator_mode=False, noise=0, termination: pnl.Condition = pnl.AllHaveRun(),
                               learning=False):
        self._switch_integrator_mode(integrator_mode)
        self._switch_noise(noise)
        self.system.termination_processing = {pnl.TimeScale.TRIAL: termination}
        for learning_mechanism, learning_rate in self.learning_mechanisms_to_learning_rates.items():
            learning_mechanism.learning_rate = learning and learning_rate or 0

    def _switch_integrator_mode(self, new_mode=False):
        for mechanism in self.integrating_mechanisms:
            mechanism.integrator_mode = new_mode

    def _switch_noise(self, noise=0):
        for mechanism in self.integrating_mechanisms:
            mechanism.noise = noise

    def _create_control_inputs(self, num_inputs_per_stimulus):
        """
        In the control condition, the primary input is set to one feature at a time, while the secondary input is
        always zero
        :param num_inputs_per_stimulus: how many inputs to create for each stimulus (feature)
        :return: two numpy arrays - one for the primary inputs, and another for the secondary inputs
        """
        total_trials = num_inputs_per_stimulus * self.num_features

        secondary_inputs = np.zeros((total_trials, self.num_features))

        primary_input_list = []
        for feature_index in range(self.num_features):
            primary_feature_inputs = np.zeros((num_inputs_per_stimulus, self.num_features))
            primary_feature_inputs[:, feature_index] = 1
            primary_input_list.append(primary_feature_inputs)

        primary_inputs = np.concatenate(primary_input_list)
        return primary_inputs, secondary_inputs

    def _create_congruent_inputs(self, num_inputs_per_stimulus):
        """
        In the congruent condition, the primary input is again set to a single feature at a time, and the secondary
        input is set to the same input as the primary one
        :param num_inputs_per_stimulus: how many inputs to create for each stimulus (feature)
        :return: two numpy arrays - one for the primary inputs, and another for the secondary inputs
        """
        primary_input_list = []
        for feature_index in range(self.num_features):
            primary_feature_inputs = np.zeros((num_inputs_per_stimulus, self.num_features))
            primary_feature_inputs[:, feature_index] = 1
            primary_input_list.append(primary_feature_inputs)

        primary_inputs = np.concatenate(primary_input_list)
        secondary_inputs = np.copy(primary_inputs)
        return primary_inputs, secondary_inputs

    def _create_conflict_inputs(self, num_inputs_per_stimulus):
        """
        In the conflict condition, the primary input is again set to a single feature at a time, and the secondary
        input is set to one of the other (incongruent) features
        :param num_inputs_per_stimulus: how many inputs to create for each stimulus (feature)
        :return: two numpy arrays - one for the primary inputs, and another for the secondary inputs
        """
        primary_input_list = []
        secondary_input_list = []
        for primary_feature_index in range(self.num_features):
            primary_feature_inputs = np.zeros((num_inputs_per_stimulus, self.num_features))
            primary_feature_inputs[:, primary_feature_index] = 1
            primary_input_list.append(primary_feature_inputs)

            secondary_feature_inputs = np.zeros((num_inputs_per_stimulus, self.num_features))
            conflict_feature_indices = np.arange(self.num_features)
            conflict_feature_indices = np.delete(conflict_feature_indices, primary_feature_index)
            num_conflict_features = self.num_features - 1

            for i in range(num_conflict_features):
                conflict_feature_index = conflict_feature_indices[i]
                inputs_per_conflict_feature = num_inputs_per_stimulus / num_conflict_features
                start = int(np.round(i * inputs_per_conflict_feature))
                end = int(np.round((i + 1) * inputs_per_conflict_feature))
                secondary_feature_inputs[start:end, conflict_feature_index] = 1

            secondary_input_list.append(secondary_feature_inputs)

        primary_inputs = np.concatenate(primary_input_list)
        secondary_inputs = np.concatenate(secondary_input_list)
        return primary_inputs, secondary_inputs

    def _create_train_inputs(self, randomize_order, trials_per_stimulus_per_train_block):

        trials_per_task_per_block = trials_per_stimulus_per_train_block * self.num_features

        # Permute the inputs and targets
        if randomize_order:
            perm = np.random.permutation(trials_per_task_per_block)
        else:
            perm = range(trials_per_task_per_block)

        # Create all inputs and targets
        on_task_inputs = np.ones((trials_per_task_per_block,))
        off_task_inputs = np.zeros((trials_per_task_per_block,))

        first_task_inputs = np.concatenate((on_task_inputs, off_task_inputs))[perm]
        second_task_inputs = np.concatenate((off_task_inputs, on_task_inputs))[perm]

        on_inputs, off_inputs = self._create_control_inputs(trials_per_stimulus_per_train_block)

        first_inputs = np.concatenate((on_inputs, off_inputs))[perm]
        second_inputs = np.concatenate((off_inputs, on_inputs))[perm]

        targets = np.concatenate((on_inputs, on_inputs))[perm]

        # Create I/O dictionaries
        input_dict = {
            self._first_pathway_input_layer: first_inputs,
            self._second_pathway_input_layer: second_inputs,
            self._first_pathway_task_layer: first_task_inputs,
            self._second_pathway_task_layer: second_task_inputs
        }

        target_dict = {self.output_layer: targets}

        return input_dict, target_dict

    def train(self, trials_per_stimulus_per_train_block=STROOP_TRAIN_TRIALS_PER_STIMULUS, randomize_order=True):
        """
        Train the stroop-naming model, according to the specification in Cohen et al. (1990).

        TODO: Laura should add ratio parameters to _create_train_inputs to control the ratio between how much
        training each task receives relative to each other

        :return:
        """
        total_trials_per_block = trials_per_stimulus_per_train_block * self.num_features
        input_dict, target_dict = self._create_train_inputs(randomize_order, trials_per_stimulus_per_train_block)

        # Set in default mode - no integration, no noise, termination is pnl.AllHaveRun()
        self._switch_trial_settings(learning=True)

        return self.system.run(inputs=input_dict, targets=target_dict)[-total_trials_per_block:]

    def _extract_rt_cycles(self):
        # Grab the log dictionary from the output layer
        log_dict = self.output_layer.log.nparray_dictionary()

        # Extract out the relevant keys from the log to a single numpy array
        relevant_key_arrays = [np.array([x[0] for x in log_dict[key]]) for key in ('Run', 'Trial', 'Pass')]
        table = np.stack(relevant_key_arrays, axis=1)

        # Filter out only the last run
        last_run = np.max(table[:, 0])
        table = table[table[:, 0] == last_run]

        # Filter out only the last pass of each trial
        trial_ends = (table[1:, 1] - table[:-1, 1]) != 0
        trial_ends = np.append(trial_ends, True)
        last_passes = table[trial_ends, :]

        # Filter out only odd trials
        last_passes = last_passes[last_passes[:, 1] % 2 == 1, :]
        return last_passes[:, 2]

    def _test_single_phase(self, primary_test_inputs, secondary_test_inputs, conditions, primary_input_layer,
                           secondary_input_layer, primary_task_layer, secondary_task_layer):

        # Create a copy of the inputs, randomly permute the order
        total_num_inputs = primary_test_inputs.shape[0]
        input_dict, target_dict, shuffled_conditions = self._create_test_inputs(conditions, primary_input_layer,
                                                                                primary_task_layer, primary_test_inputs,
                                                                                secondary_input_layer,
                                                                                secondary_task_layer,
                                                                                secondary_test_inputs)

        # start off in the initialization trial mode
        self._switch_trial_settings()

        def switch():
            self._switch_trial_type()

        # TODO: do I want to do anything with the outputs proper?
        outputs = self.system.run(inputs=input_dict, targets=target_dict, call_after_trial=switch)[-total_num_inputs:]

        # TODO: if I care about how correct the model was in each case, I can grab the accumulator values similarly to
        # TODO: how I grab the RT cycles, and compare which accumulator hit condition to the target value

        rt_cycles = self._extract_rt_cycles()
        results = defaultdict(list)
        for condition, rt in zip(shuffled_conditions, rt_cycles):
            results[condition].append(rt)

        return results

    def _create_test_inputs(self, conditions, primary_input_layer, primary_task_layer, primary_test_inputs,
                            secondary_input_layer, secondary_task_layer, secondary_test_inputs):

        total_num_inputs = primary_test_inputs.shape[0]
        perm = np.random.permutation(total_num_inputs)

        shuffled_primary_inputs = np.copy(primary_test_inputs)[perm]
        shuffled_secondary_inputs = np.copy(secondary_test_inputs)[perm]
        shuffled_conditions = [conditions[i] for i in perm]

        # duplicate everything twice, in order to add the initialization trials
        shuffled_primary_inputs = np.repeat(shuffled_primary_inputs, 2, axis=0)
        shuffled_secondary_inputs = np.repeat(shuffled_secondary_inputs, 2, axis=0)

        # zero out the inputs on every other trial, for the task initialization ones
        shuffled_primary_inputs[::2, :] = 0
        shuffled_secondary_inputs[::2, :] = 0

        # Create the task inputs, duplicating to account for duplication above
        primary_task_inputs = np.ones((total_num_inputs * 2,))
        secondary_task_inputs = np.zeros((total_num_inputs * 2,))
        targets = np.copy(shuffled_primary_inputs)

        # Create I/O dictionaries
        input_dict = {
            primary_input_layer: shuffled_primary_inputs,
            secondary_input_layer: shuffled_secondary_inputs,
            primary_task_layer: primary_task_inputs,
            secondary_task_layer: secondary_task_inputs
        }

        target_dict = {self.output_layer: targets}

        return input_dict, target_dict, shuffled_conditions

    def last_run_to_dataframe(self):
        dataframes = []
        first = True
        for log_layer in self.log_layers:
            layer_size = log_layer.size[0]
            log_dict = log_layer.log.nparray_dictionary()

            # Extract out all keys, treating value specially since it's already an np array
            arrays = [np.array([x[0] for x in log_dict[key]]) for key in ('Run', 'Trial', 'Pass', 'Time_step')]
            arrays.extend([np.squeeze(log_dict['value'][:, :, i]) for i in range(layer_size)])
            table = np.stack(arrays, axis=1)

            # Filter out only the last run
            last_run = np.max(table[:, 0])
            table = table[table[:, 0] == last_run]

            # Create as dataframe and add to the list of dataframes
            if first:
                df = pandas.DataFrame(table, columns=['Run', 'Trial', 'Pass', 'Time_step'] +
                                                     [f'{log_layer.name}_{i}' for i in range(layer_size)])
                first = False

            else:
                df = pandas.DataFrame(table[:, -1 * layer_size:], columns=[f'{log_layer.name}_{i}'
                                                                           for i in range(layer_size)])

            dataframes.append(df)

        return pandas.concat(dataframes, axis=1, join='inner')

    def test(self, trials_per_stimulus_per_test_block=STROOP_TEST_TRIALS_PER_STIMULUS, randomize_phase_order=True,
             control_condition=True, congruent_condition=True, conflict_condition=True):
        """
        See the long comment in the train method. This implements a single testing block, for both types of task.
        :param trials_per_stimulus_per_test_block:
        :param randomize_phase_order:
        :return:
        """
        num_conditions = int(control_condition) + int(congruent_condition) + int(conflict_condition)
        inputs_per_condition = trials_per_stimulus_per_test_block // num_conditions

        primary_inputs = []
        secondary_inputs = []
        conditions = []

        for condition_flag, create_func, condition_name in zip(
                (control_condition, congruent_condition, conflict_condition),
                (self._create_control_inputs, self._create_congruent_inputs, self._create_conflict_inputs),
                (CONTROL_CONDITION, CONGRUENT_CONDITION, CONFLICT_CONDITION)):

            if condition_flag:
                primary, secondary = create_func(inputs_per_condition)
                primary_inputs.append(primary)
                secondary_inputs.append(secondary)
                conditions.extend([condition_name] * inputs_per_condition * self.num_features)

        primary_test_inputs = np.concatenate(primary_inputs)
        secondary_test_inputs = np.concatenate(secondary_inputs)

        def test_first_pathway():
            results = self._test_single_phase(primary_test_inputs, secondary_test_inputs, conditions,
                                              self._first_pathway_input_layer, self._second_pathway_input_layer,
                                              self._first_pathway_task_layer, self._second_pathway_task_layer)
            self.last_first_pathway_test_df = self.last_run_to_dataframe()
            return results

        def test_second_pathway():
            results = self._test_single_phase(primary_test_inputs, secondary_test_inputs, conditions,
                                              self._second_pathway_input_layer, self._first_pathway_input_layer,
                                              self._second_pathway_task_layer, self._first_pathway_task_layer)
            self.last_second_pathway_test_df = self.last_run_to_dataframe()
            return results

        # run in one order
        if randomize_phase_order and random.uniform() < 0.5:
            first_pathway_results = test_first_pathway()
            second_pathway_results = test_second_pathway()
            first_pathway_tested_first = True

        # run in the opposite order
        else:
            second_pathway_results = test_second_pathway()
            first_pathway_results = test_first_pathway()
            first_pathway_tested_first = False

        return first_pathway_results, second_pathway_results, first_pathway_tested_first

