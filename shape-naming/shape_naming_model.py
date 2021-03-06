import numpy as np
from numpy import random
import psyneulink as pnl
from collections import defaultdict
from scipy import io
import pandas


# Setting up default network parameters
#  TODO: find the correct values for all of these
DEFAULT_NUM_FEATURES = 2
DEFAULT_HIDDEN_LAYER_SIZE = 2
DEFAULT_LEARNING_RATE = 0.02
DEFAULT_HIDDEN_BIAS = 4  # PNL bias is subtracted, so this mean a bias of -4
DEFAULT_HIDDEN_GAIN = 1
DEFAULT_INTEGRATION_RATE = 0.1
DEFAULT_INTEGRATOR_MODE = True
DEFAULT_NOISE_STD = 0.01

DEFAULT_INDIRECT_LAYER_SIZE = 2
DEFAULT_INDIRECT_LEARNING_RATE = 0.2
DEFAULT_INDIRECT_BIAS = 2
DEFAULT_INDIRECT_LOG_CONDITIONS = 'value'

DEFAULT_ACCUMULATOR_RATE = 0.2
DEFAULT_ACCUMULATOR_NOISE_STD = 0.01
DEFAULT_ACCUMULATOR_THRESHOLD = 1.0

DEFAULT_WEIGHT_INIT_SCALE = 2e-2

DEFAULT_NAME = 'Shape-Naming'


# Loaded weight parameters
COLOR_HIDDEN_KEY = 'weightsColorHidden'
COLOR_TASK_HIDDEN_KEY = 'weightsTaskColorHidden'

SHAPE_HIDDEN_KEY = 'weightsShapeHidden'
SHAPE_TASK_HIDDEN_KEY = 'weightsTaskShapeHidden'

COLOR_OUTPUT_KEY = 'weightsColorOutput'
SHAPE_OUTPUT_KEY = 'weightsShapeOutput'

SHAPE_INDIRECT_KEY = 'weightsShapeIndirect'
INDIRECT_OUTPUT_KEY = 'weightsIndirectOutput'

FIRST_ACCUMULATOR_DIFFERENCING_KEY = 'firstAccumulatorDifferencing'
SECOND_ACCUMULATOR_DIFFERENCING_KEY = 'secondAccumulatorDifferencing'

"""
Shape to hidden unites are described as "initial connection strengths from the input to the intermediate units
that allowed it to generate a useful representation at the level of the intermediate units" -- 
so maybe a bit weaker than color?

Words are 2.6, colors 2.2, so let's try 1.8?
"""

DEFAULT_WEIGHT_DICT = {
    COLOR_HIDDEN_KEY: np.matrix([[2.2, -2.2],
                                 [-2.2, 2.2]]),
    # TODO: Given the comment above, what should these be?
    SHAPE_HIDDEN_KEY: np.matrix([[1.8, -1.8],
                                 [-1.8, 1.8]]),
    COLOR_TASK_HIDDEN_KEY: np.matrix([[4.0, 4.0]]),
    SHAPE_TASK_HIDDEN_KEY: np.matrix([[4.0, 4.0]]),
    COLOR_OUTPUT_KEY: np.matrix([[1.3, -1.3],
                                 [-1.3, 1.3]]),
    # No shape => output weights - these are learned: "small random strengths were
    # assigned to the connections between the intermediate and output units."
    FIRST_ACCUMULATOR_DIFFERENCING_KEY: np.matrix([[1.0], [-1.0]]),
    SECOND_ACCUMULATOR_DIFFERENCING_KEY: np.matrix([[-1.0], [1.0]])
}

# Training-related constants
DEFAULT_NUM_TRIALS_PER_STIMULUS = 72
CONTROL_CONDITION = 'control'
CONGRUENT_CONDITION = 'congruent'
CONFLICT_CONDITION = 'conflict'


class ShapeNamingModel:
    def __init__(self, num_features=DEFAULT_NUM_FEATURES, indirect_path=True,
                 weight_file=None, weight_dict=DEFAULT_WEIGHT_DICT, log_values=True, *,
                 hidden_layer_size=DEFAULT_HIDDEN_LAYER_SIZE, hidden_bias=DEFAULT_HIDDEN_BIAS,
                 hidden_gain=DEFAULT_HIDDEN_GAIN, integration_rate=DEFAULT_INTEGRATION_RATE,
                 integrator_mode=DEFAULT_INTEGRATOR_MODE, noise_std=DEFAULT_NOISE_STD,
                 direct_learning_rate=DEFAULT_LEARNING_RATE, indirect_layer_size=DEFAULT_INDIRECT_LAYER_SIZE,
                 indirect_learning_rate=DEFAULT_INDIRECT_LEARNING_RATE, indirect_bias=DEFAULT_INDIRECT_BIAS,
                 indirect_log_conditions=DEFAULT_INDIRECT_LOG_CONDITIONS,
                 accumulator_rate=DEFAULT_ACCUMULATOR_RATE, accumulator_noise_std=DEFAULT_ACCUMULATOR_NOISE_STD,
                 accumulator_threshold=DEFAULT_ACCUMULATOR_THRESHOLD, weight_init_scale=DEFAULT_WEIGHT_INIT_SCALE,
                 name=DEFAULT_NAME):

        self.num_features = num_features
        self.indirect_path = indirect_path
        # TODO: verify if I even need this parameter anymore?
        # self.learning = learning
        self.log_values = log_values

        if weight_file is not None and weight_dict is not None:
            raise ValueError('Cannot provide both weight file and weight dictionary')

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
        self.direct_learning_rate = direct_learning_rate
        self.integration_rate = integration_rate
        self.integrator_mode = integrator_mode
        self.noise_std = noise_std
        self.hidden_bias = hidden_bias
        self.hidden_gain = hidden_gain

        self.indirect_layer_size = indirect_layer_size
        self.indirect_learning_rate = indirect_learning_rate
        self.indirect_bias = indirect_bias
        self.indirect_log_conditions = indirect_log_conditions

        self.accumulator_rate = accumulator_rate
        self.accumulator_noise_std = accumulator_noise_std
        self.accumulator_threshold = accumulator_threshold

        self.weight_init_scale = weight_init_scale
        self.name = name

        self._generate_layers()
        self._generate_processes()
        self._generate_system()

    def _generate_layers(self):
        """
        Generate the layers for this model. The hidden layers use an integrator mode, rate, and noise function.
        TODO: does the indirect pathway accumulate exactly the same as the hidden and output layers?
        :return: None, saves the layers into a whole bunch of members
        """
        # Inputs
        self.color_input_layer = pnl.TransferMechanism(size=self.num_features, name='color_input')
        self.shape_input_layer = pnl.TransferMechanism(size=self.num_features, name='shape_input')

        # Task units
        self.color_task_layer = pnl.TransferMechanism(size=1, name='color_task')
        self.shape_task_layer = pnl.TransferMechanism(size=1, name='shape_task')

        # Hidden layers
        self.color_hidden_layer = pnl.TransferMechanism(size=self.hidden_layer_size,
                                                        name='color_hidden',
                                                        function=pnl.Logistic(gain=self.hidden_gain,
                                                                              bias=self.hidden_bias),
                                                        integrator_mode=self.integrator_mode,
                                                        integration_rate=self.integration_rate,
                                                        noise=self._generate_noise_function())
        self.shape_hidden_layer = pnl.TransferMechanism(size=self.hidden_layer_size,
                                                        name='shape_hidden',
                                                        function=pnl.Logistic(gain=self.hidden_gain,
                                                                              bias=self.hidden_bias),
                                                        integrator_mode=self.integrator_mode,
                                                        integration_rate=self.integration_rate,
                                                        noise=self._generate_noise_function())

        # self.color_dummy = pnl.TransferMechanism(size=self.hidden_layer_size, name='dummy')

        if self.indirect_path:
            self._generate_indirect_layer()

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

        if self.log_values:
            self.log_layers = [self.color_hidden_layer, self.shape_hidden_layer,
                               self.output_layer, self.first_accumulator, self.second_accumulator]

            for layer in self.log_layers:
                layer.set_log_conditions('value')

            if self.indirect_path:
                # Inserting there for it to appear in the correct order in output dataframe
                self.log_layers.insert(2, self.indirect_shape_layer)
                self.indirect_shape_layer.set_log_conditions(self.indirect_log_conditions)

    def _generate_indirect_layer(self):
        self.indirect_shape_layer = pnl.TransferMechanism(size=self.indirect_layer_size,
                                                          name='shape_indirect',
                                                          function=pnl.Logistic(bias=self.indirect_bias),
                                                          integrator_mode=self.integrator_mode,
                                                          integration_rate=self.integration_rate,
                                                          noise=self._generate_noise_function())

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

        self.color_hidden_process = self._generate_process(COLOR_HIDDEN_KEY,
                                                           self.num_features, self.hidden_layer_size,
                                                           self.color_input_layer, self.color_hidden_layer,
                                                           'color-hidden-proc')

        self.color_task_hidden_process = self._generate_process(COLOR_TASK_HIDDEN_KEY,
                                                                1, self.hidden_layer_size,
                                                                self.color_task_layer, self.color_hidden_layer,
                                                                'task-color-hidden-proc')

        self.shape_hidden_process = self._generate_process(SHAPE_HIDDEN_KEY,
                                                           self.num_features, self.hidden_layer_size,
                                                           self.shape_input_layer, self.shape_hidden_layer,
                                                           'shape-hidden-proc')

        self.shape_task_hidden_process = self._generate_process(SHAPE_TASK_HIDDEN_KEY,
                                                                1, self.hidden_layer_size,
                                                                self.shape_task_layer, self.shape_hidden_layer,
                                                                'task-shape-hidden-proc')

        self.color_output_process = self._generate_process(COLOR_OUTPUT_KEY,
                                                           self.hidden_layer_size, self.num_features,
                                                           self.color_hidden_layer, self.output_layer,
                                                           'color-output-proc')

        # If this moves below the shape_output_process, everything goes to hell. Why?
        if self.indirect_path:
            self._generate_indirect_processes()

        self.shape_output_process = self._generate_process(SHAPE_OUTPUT_KEY,
                                                           self.hidden_layer_size, self.num_features,
                                                           self.shape_hidden_layer, self.output_layer,
                                                           'shape-output-proc', self.direct_learning_rate)

        self.first_accumulator_process = self._generate_process(FIRST_ACCUMULATOR_DIFFERENCING_KEY,
                                                                self.num_features, self.num_features,
                                                                self.output_layer, self.first_accumulator,
                                                                'first-accumulator-process')

        self.second_accumulator_process = self._generate_process(SECOND_ACCUMULATOR_DIFFERENCING_KEY,
                                                                 self.num_features, self.num_features,
                                                                 self.output_layer, self.second_accumulator,
                                                                 'second-accumulator-process')

        self.integrating_mechanisms = [self.color_hidden_layer, self.shape_hidden_layer, self.output_layer]
        if self.indirect_path:
            self.integrating_mechanisms.append(self.indirect_shape_layer)

    def _generate_indirect_processes(self):
        self.shape_indirect_process = self._generate_process(SHAPE_INDIRECT_KEY,
                                                             self.hidden_layer_size, self.indirect_layer_size,
                                                             self.shape_hidden_layer, self.indirect_shape_layer,
                                                             'shape-indirect-proc', self.indirect_learning_rate)
        self.indirect_output_process = self._generate_process(INDIRECT_OUTPUT_KEY,
                                                              self.indirect_layer_size, self.num_features,
                                                              self.indirect_shape_layer, self.output_layer,
                                                              'indirect-output-proc', self.indirect_learning_rate)

    def _generate_system(self):
        """
        Generate the overall PNL system for this model. Given that all of the heavy lifting is done above,
        this doesn't really do anything.
        :return: None; PsyNeuLink system generated.
        """
        # Adding learning rates to processes, as they have different ones, rather than here
        self.system = pnl.System(
            name=self.name,
            processes=self.processes,
            # learning_rate=self.learning_rate
        )

    def _generate_noise_function(self):
        """
        Generate the noise function with the supplied `noise_std`, split to a member since this tends to recurr.
        :return: A PsyNeuLink noise function with a normal noise distribution
        """
        return pnl.NormalDist(standard_dev=self.noise_std).function

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

    def train(self, trials_per_stimulus_per_train_block=DEFAULT_NUM_TRIALS_PER_STIMULUS, randomize_order=True):
        """
        Train the shape-naming model, according to the specification in Cohen et al. (1990).

        To the extent I understand it right now:
        * We train on independent (shape-only) stimuli, with the task unit for shape on
        * One day corresponds to 72 trials per stimulus, five days to 504 trials per stimulus, and 20 days to 2520?
        * The original shape-naming paper, MacLeod & Dunbar (1988), defines four phases, only some of which are relevant:
            1. Color-naming baseline. These are irrelevant here, since the network is trained on colors already.
            2. Shape-naming learning. Let's assume these are 72 trials/stimulus with no color presented, shape task on.
            3. Color-naming testing. In the Cohen et al. paper this is discussed as being done in both the congruent
            (shape and color pointing to same out), conflict (shape and color to different output), and control (neutral
            shape, which corresponds to no input. Let's assume this comes out to another 72/stimulus somehow?
            4. Shape-naming testing. Again split congruent/control/conflict. We can assume 72/stimulus again?
        * Cohen et al. (1990) mentions that subjects received feedback during the test trials, so the network was
        allowed to learn from them as well. Presumably this still only holds for the shape hidden => output weights?
        * Let's see how the math works out:
            * Day 1 gets 72 trials/stimulus of training, and then 72*2=144/stimulus of testing.
            * Days 2-4 gets 72 trials/stimulus of training.
            * Day 5 gets another 72 t/s of training, for a total of 72*7=504 pre-testing stimulus. This is followed by
            another 72*2=144 trials of testing.
            * So far this comes out to a total of 9 72-t/s blocks. By the beginning of the day 25 test, Cohen et al.
            report a total of 2520 t/s = 35 * 72. It is unclear how we make up another 28 72-trial blocks in the
            15 days that remain.

        The original MacLeod & Dunbar (1988) paper is here:
        https://pdfs.semanticscholar.org/fd0c/e33748089c1c1f1a1df746572844dc292243.pdf

        TODO: do the trial numbers include the testing trials? Or are they before testing, as I interpreted them above?

        I think, though, is that a training block is 72 trials per stimuli of shape-naming in the control
        condition, without trying to record RTs or anything, and a testing block is 72 trials/stimuli split over the
        conflict, congruent, and control conditions, for both tasks.

        :return:
        """
        total_trials_per_block = trials_per_stimulus_per_train_block * self.num_features
        input_dict, target_dict = self._create_train_inputs(randomize_order, trials_per_stimulus_per_train_block)

        # Set in default mode - no integration, no noise, termination is pnl.AllHaveRun()
        self._switch_trial_settings(learning=True)

        return self.system.run(inputs=input_dict, targets=target_dict)[-total_trials_per_block:]

    def _create_train_inputs(self, randomize_order, trials_per_stimulus_per_train_block):

        total_trials_per_block = trials_per_stimulus_per_train_block * self.num_features

        # Create all inputs and targets
        shape_task_inputs = np.ones((total_trials_per_block,))
        color_task_inputs = np.zeros((total_trials_per_block,))
        shape_inputs, color_inputs = self._create_control_inputs(trials_per_stimulus_per_train_block)
        targets = np.copy(shape_inputs)

        # Permute the inputs and targets
        if randomize_order:
            perm = np.random.permutation(total_trials_per_block)
        else:
            perm = range(total_trials_per_block)

        shape_inputs = shape_inputs[perm]
        targets = targets[perm]

        # Create I/O dictionaries
        input_dict = {
            self.color_task_layer: color_task_inputs,
            self.shape_task_layer: shape_task_inputs,
            self.color_input_layer: color_inputs,
            self.shape_input_layer: shape_inputs
        }

        # TODO: do I compare to the accumulator values? Or the output layer?
        target_dict = {self.output_layer: targets}

        return input_dict, target_dict

    def test(self, trials_per_stimulus_per_test_block=DEFAULT_NUM_TRIALS_PER_STIMULUS, randomize_phase_order=True,
             control_condition=True, congruent_condition=True, conflict_condition=True):
        """
        See the long comment in the train method. This implements a single testing block, for both types of task.
        :param trials_per_stimulus_per_test_block:
        :param randomize_phase_order:
        :return:
        """
        # TODO: verify the exact numbers here
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

        def test_shape_naming():
            results = self._test_single_phase(primary_test_inputs, secondary_test_inputs, conditions,
                                              self.shape_input_layer, self.color_input_layer,
                                              self.shape_task_layer, self.color_task_layer)
            self.last_shape_naming_df = self.last_run_to_dataframe()
            return results

        def test_color_naming():
            results = self._test_single_phase(primary_test_inputs, secondary_test_inputs, conditions,
                                              self.color_input_layer, self.shape_input_layer,
                                              self.color_task_layer, self.shape_task_layer)
            self.last_color_naming_df = self.last_run_to_dataframe()
            return results

        # run in one order
        if randomize_phase_order and random.uniform() < 0.5:
            shape_naming_results = test_shape_naming()
            color_naming_results = test_color_naming()
            run_shape_naming_first = True

        # run in the opposite order
        else:
            color_naming_results = test_color_naming()
            shape_naming_results = test_shape_naming()
            run_shape_naming_first = False

        return shape_naming_results, color_naming_results, run_shape_naming_first

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

        # TODO: do I compare to the accumulator values? Or the output layer?
        target_dict = {self.output_layer: targets}

        return input_dict, target_dict, shuffled_conditions

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


HEBBIAN_INDIRECT_LOG_CONDITIONS = ['value', 'matrix', 'max_passes']


class HebbianShapeNamingModel(ShapeNamingModel):
    def __init__(self, num_features=DEFAULT_NUM_FEATURES, indirect_path=True,
                 weight_file=None, weight_dict=DEFAULT_WEIGHT_DICT, log_values=True, *,
                 hidden_layer_size=DEFAULT_HIDDEN_LAYER_SIZE, hidden_bias=DEFAULT_HIDDEN_BIAS,
                 hidden_gain=DEFAULT_HIDDEN_GAIN, integration_rate=DEFAULT_INTEGRATION_RATE,
                 integrator_mode=DEFAULT_INTEGRATOR_MODE, noise_std=DEFAULT_NOISE_STD,
                 direct_learning_rate=DEFAULT_LEARNING_RATE, indirect_layer_size=DEFAULT_INDIRECT_LAYER_SIZE,
                 indirect_learning_rate=DEFAULT_INDIRECT_LEARNING_RATE, indirect_bias=DEFAULT_INDIRECT_BIAS,
                 indirect_log_conditions=HEBBIAN_INDIRECT_LOG_CONDITIONS,
                 accumulator_rate=DEFAULT_ACCUMULATOR_RATE, accumulator_noise_std=DEFAULT_ACCUMULATOR_NOISE_STD,
                 accumulator_threshold=DEFAULT_ACCUMULATOR_THRESHOLD, weight_init_scale=DEFAULT_WEIGHT_INIT_SCALE,
                 name=DEFAULT_NAME):

        super(HebbianShapeNamingModel, self).__init__(
            num_features, indirect_path, weight_file, weight_dict, log_values, hidden_layer_size=hidden_layer_size,
            hidden_bias=hidden_bias, hidden_gain=hidden_gain, integration_rate=integration_rate,
            integrator_mode=integrator_mode, noise_std=noise_std, direct_learning_rate=direct_learning_rate,
            indirect_layer_size=indirect_layer_size, indirect_learning_rate=indirect_learning_rate,
            indirect_bias=indirect_bias, indirect_log_conditions=indirect_log_conditions,
            accumulator_rate=accumulator_rate, accumulator_noise_std=accumulator_noise_std,
            accumulator_threshold=accumulator_threshold, weight_init_scale=weight_init_scale, name=name)

    def _generate_indirect_layer(self):
        # TODO: initial weights - does a hollow matrix make sense? Yes, but with smaller initial values
        matrix_size = 2 * self.num_features + self.indirect_layer_size
        matrix = pnl.random_matrix(matrix_size, matrix_size, 2, -1) * self.weight_init_scale
        np.fill_diagonal(matrix, 0)

        # TODO: why is the parameter here called enable_learning rather than learning?
        self.indirect_shape_layer = pnl.ContrastiveHebbianMechanism(
            input_size=self.num_features, hidden_size=self.indirect_layer_size, target_size=self.num_features,
            separated=True, matrix=matrix,
            integrator_mode=self.integrator_mode, integration_rate=self.integration_rate,
            max_passes=1000,
            noise=self._generate_noise_function(),
            learning_rate=self.indirect_learning_rate,
            enable_learning=self.indirect_learning_rate != 0
        )

        self.indirect_target_input = pnl.TransferMechanism(size=self.num_features, name='chl-target-input')

    def _generate_indirect_processes(self):
        learning = self.indirect_learning_rate != 0
        self.shape_indirect_process = pnl.Process(
            pathway=[self.shape_hidden_layer, self.indirect_shape_layer, self.output_layer],
            name='shape-chl-process')
        self.processes.append(self.shape_indirect_process)

        if learning:
            learning_mechanism = self.indirect_shape_layer.learning_mechanism
            learning_mechanism.learning_rate = self.indirect_learning_rate
            self.learning_mechanisms_to_learning_rates[learning_mechanism] = self.indirect_learning_rate

        target_to_chl_projection = pnl.MappingProjection(sender=self.indirect_target_input.output_states[0],
                                                         receiver=self.indirect_shape_layer.input_states[2])
        self.indirect_target_input_process = pnl.Process(
            pathway=[self.indirect_target_input, target_to_chl_projection, self.indirect_shape_layer],
            name='chl-target-process'
        )
        self.processes.append(self.indirect_target_input_process)

    def _create_train_inputs(self, randomize_order, trials_per_stimulus_per_train_block):
        input_dict, target_dict, = \
            super(HebbianShapeNamingModel, self)._create_train_inputs(randomize_order,
                                                                      trials_per_stimulus_per_train_block)

        input_dict[self.indirect_target_input] = target_dict[self.output_layer]
        return input_dict, target_dict

    def _create_test_inputs(self, conditions, primary_input_layer, primary_task_layer, primary_test_inputs,
                            secondary_input_layer, secondary_task_layer, secondary_test_inputs):
        input_dict, target_dict, shuffled_conditions = \
            super(HebbianShapeNamingModel, self)._create_test_inputs(
                conditions, primary_input_layer, primary_task_layer, primary_test_inputs,
                secondary_input_layer, secondary_task_layer, secondary_test_inputs)

        input_dict[self.indirect_target_input] = target_dict[self.output_layer]
        return input_dict, target_dict, shuffled_conditions

