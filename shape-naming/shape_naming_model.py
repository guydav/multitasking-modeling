import numpy as np
from numpy import random
import psyneulink as pnl
from sklearn.metrics import mean_squared_error
from scipy import io
import time



# Setting up default network parameters
#  TODO: find the correct values for all of these
DEFAULT_HIDDEN_LAYER_SIZE = 2
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_HIDDEN_BIAS = 4  # PNL bias is subtracted, so this mean a bias of -4
DEFAULT_HIDDEN_GAIN = 1
DEFAULT_INTEGRATION_RATE = 0.1
DEFAULT_INTEGRATOR_MODE = True
DEFAULT_NOISE_STD = 0.05

DEFAULT_INDIRECT_LAYER_SIZE = 2
DEFAULT_INDIRECT_LEARNING_RATE = 0.5
DEFAULT_FAST_BIAS = 0

DEFAULT_ACCUMULATOR_RATE = 0.1
DEFAULT_ACCUMULATOR_NOISE_STD = 0.1
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

SHAPE_FAST_KEY = 'weightsShapeFast'
FAST_OUTPUT_KEY = 'weightsFastOutput'

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
    def __init__(self, num_features, indirect_path=True, weight_file=None, weight_dict=DEFAULT_WEIGHT_DICT,
                 learning=pnl.LEARNING, log_values=True, *, hidden_layer_size=DEFAULT_HIDDEN_LAYER_SIZE,
                 hidden_bias=DEFAULT_HIDDEN_BIAS, hidden_gain=DEFAULT_HIDDEN_GAIN,
                 integration_rate=DEFAULT_INTEGRATION_RATE, integrator_mode=DEFAULT_INTEGRATOR_MODE,
                 noise_std=DEFAULT_NOISE_STD, direct_learning_rate=DEFAULT_LEARNING_RATE,
                 indirect_layer_size=DEFAULT_INDIRECT_LAYER_SIZE, indirect_learning_rate=DEFAULT_INDIRECT_LEARNING_RATE,
                 fast_bias=DEFAULT_FAST_BIAS, accumulator_rate=DEFAULT_ACCUMULATOR_RATE,
                 accumulator_noise_std=DEFAULT_ACCUMULATOR_NOISE_STD,
                 accumulator_threshold=DEFAULT_ACCUMULATOR_THRESHOLD, weight_init_scale=DEFAULT_WEIGHT_INIT_SCALE,
                 name=DEFAULT_NAME):

        self.num_features = num_features
        self.indirect_path = indirect_path
        # TODO: verify if I even need this parameter anymore?
        self.learning = learning
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
        self.fast_bias = fast_bias

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
        if self.indirect_path:
            self.indirect_shape_layer = pnl.TransferMechanism(size=self.indirect_layer_size,
                                                              name='fast_shape',
                                                              function=pnl.Logistic(bias=self.fast_bias),
                                                              integrator_mode=self.integrator_mode,
                                                              integration_rate=self.integration_rate,
                                                              noise=self._generate_noise_function()
                                                              )

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
            for layer in (self.color_hidden_layer, self.shape_hidden_layer,
                          self.output_layer, self.first_accumulator, self.second_accumulator):
                layer.set_log_conditions('value')

    def _generate_noise_function(self):
        """
        Generate the noise function with the supplied `noise_std`, split to a member since this tends to recurr.
        :return: A PsyNeuLink noise function with a normal noise distribution
        """
        return pnl.NormalDist(standard_dev=self.noise_std).function

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

        self.integrating_mechanisms = (self.color_hidden_layer, self.shape_hidden_layer, self.output_layer)

        if self.indirect_path:
            self.shape_indirect_process = self._generate_process(SHAPE_FAST_KEY,
                                                                 self.hidden_layer_size, self.indirect_layer_size,
                                                                 self.shape_hidden_layer, self.indirect_shape_layer,
                                                                 'shape-indirect-proc', self.indirect_learning_rate)

            self.indirect_output_process = self._generate_process(FAST_OUTPUT_KEY,
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

    def _switch_trial_type(self):
        if isinstance(self.system.termination_processing[pnl.TimeScale.TRIAL], pnl.AllHaveRun):
            self._switch_to_integration_trial()
        else:
            self._switch_trial_settings()

    def _switch_to_integration_trial(self):
        def pass_threshold(first_mechanism, second_mechanism, threshold):
            # TODO: verify these are indeed numpy arrays
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

    def _create_conflict_stimuli(self, num_inputs_per_stimulus):
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
            np.delete(conflict_feature_indices, primary_feature_index)
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

        # Create all inputs and targets
        # TODO: do I need the second explicit dimension here?
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
        # TODO: add the target here as a separate input to the CHL when working with it
        input_dict = {
            self.color_task_layer: color_task_inputs,
            self.shape_task_layer: shape_task_inputs,
            self.color_input_layer: color_inputs,
            self.shape_input_layer: shape_inputs
        }
        # TODO: do I compare to the accumulator values? Or the output layer?
        target_dict = {self.output_layer: targets}

        # Set in default mode - no integration, no noise, termination is pnl.AllHaveRun()
        self._switch_trial_settings(learning=True)

        # TODO: verify learning runs, and that it only runs on the correct mechanisms
        return self.system.run(inputs=input_dict, targets=target_dict)[-total_trials_per_block:]

    def test(self, trials_per_stimulus_per_test_block=DEFAULT_NUM_TRIALS_PER_STIMULUS, randomize_phase_order=True):
        """
        See the long comment in the train method. This implements a single testing block, for both types of task.
        :param trials_per_stimulus_per_test_block:
        :param randomize_phase_order:
        :return:
        """
        # TODO: verify the exact numbers here
        inputs_per_condition = trials_per_stimulus_per_test_block // 3
        control_primary, control_secondary = self._create_control_inputs(inputs_per_condition)
        congruent_primary, congruent_secondary = self._create_conflict_stimuli(inputs_per_condition)
        conflict_primary, conflict_secondary = self._create_conflict_stimuli(inputs_per_condition)

        primary_test_inputs = np.concatenate((control_primary, congruent_primary, conflict_primary))
        secondary_test_inputs = np.concatenate((control_secondary, congruent_secondary, conflict_secondary))
        conditions = [CONTROL_CONDITION] * inputs_per_condition * self.num_features + \
                     [CONGRUENT_CONDITION] * inputs_per_condition * self.num_features + \
                     [CONFLICT_CONDITION] * inputs_per_condition * self.num_features

        def test_shape_naming():
            return self._test_single_phase(primary_test_inputs, secondary_test_inputs, conditions,
                                           self.shape_input_layer, self.color_input_layer,
                                           self.shape_task_layer, self.color_task_layer)

        def test_color_naming():
            return self._test_single_phase(primary_test_inputs, secondary_test_inputs, conditions,
                                           self.color_input_layer, self.shape_input_layer,
                                           self.color_task_layer, self.shape_task_layer)
        # run in one order
        if randomize_phase_order and random.uniform() < 0.5:
            shape_naming_outputs = test_shape_naming()
            color_naming_outputs = test_color_naming()
            shape_naming_first = True

        # run in the opposite order
        else:
            color_naming_outputs = test_color_naming()
            shape_naming_outputs = test_shape_naming()
            shape_naming_first = False

        return shape_naming_outputs, color_naming_outputs, shape_naming_first

    def _test_single_phase(self, primary_test_inputs, secondary_test_inputs, conditions, primary_input_layer,
                           secondary_input_layer, primary_task_layer, secondary_task_layer):

        # Create a copy of the inputs, randomly permute the order
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
        # TODO: add the target here as a separate input to the CHL when working with it
        input_dict = {
            primary_input_layer: shuffled_primary_inputs,
            secondary_input_layer: shuffled_secondary_inputs,
            primary_task_layer: primary_task_inputs,
            secondary_task_layer: secondary_task_inputs
        }

        # TODO: do I compare to the accumulator values? Or the output layer?
        target_dict = {self.output_layer: targets}

        # start off in the initialization trial mode
        self._switch_trial_settings()

        def switch():
            self._switch_trial_type()

        outputs = self.system.run(inputs=input_dict, targets=target_dict, call_after_trial=switch)[-total_num_inputs:]

        # TODO: record the RTs (# cycles) for each integrator trial (ignoring initialization trials, if they exist)

        # TODO: return the per-trial RTs in some fashion, perhaps aggregating here over which task was active and
        # TODO: which condition it was (conflict/congruent/control), using the shuffled_conditions variable
        return outputs
