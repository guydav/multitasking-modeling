import numpy as np
import psyneulink as pnl
from sklearn.metrics import mean_squared_error
from scipy import io
import time


# Setting up default network parameters
#  TODO: find the correct values for all of these
DEFAULT_HIDDEN_LAYER_SIZE = 2
DEFAULT_LEARNING_RATE = 0.3
DEFAULT_HIDDEN_BIAS = 4  # PNL bias is subtracted, so this mean a bias of -4
DEFAULT_HIDDEN_GAIN = 1
DEFAULT_INTEGRATION_RATE = 0.1
DEFAULT_INTEGRATOR_MODE = True
DEFAULT_NOISE_STD = 0.05

DEFAULT_FAST_LAYER_SIZE = 2
DEFAULT_FAST_LEARNING_RATE = 1.0
DEFAULT_FAST_BIAS = 0

DEFAULT_ACCUMULATOR_RATE = 0.1
DEFAULT_ACCUMULATOR_NOISE_STD = 0.1
DEFAULT_ACCUMULATOR_THRESHOLD = 1.0

DEFAULT_WEIGHT_INIT_SCALE = 2e-2

DEFAULT_NAME = 'Shape-Naming'

# Runtime/training parameters
DEFAULT_STOPPING_THRESHOLD = 1e-4

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

DEFAULT_WEIGHT_DICT = {
    COLOR_HIDDEN_KEY: np.matrix([[2.2, -2.2],
                                 [-2.2, 2.2]]),
    # Shape to hidden unites are described as "initial
    # connection strengths from the input to the intermediate units
    # that allowed it to generate a useful representation at the level of
    # the intermediate units" -- so maybe a bit weaker than color?
    # Words are 2.6, colors 2.2, so let's try 1.8?
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


class ShapeNamingModel:
    def __init__(self, num_features, fast_path=True, weight_file=None, weight_dict=DEFAULT_WEIGHT_DICT,
                 learning=pnl.LEARNING, log_values=True, *,
                 hidden_layer_size=DEFAULT_HIDDEN_LAYER_SIZE, hidden_learning_rate=DEFAULT_LEARNING_RATE,
                 hidden_bias=DEFAULT_HIDDEN_BIAS, hidden_gain=DEFAULT_HIDDEN_GAIN,
                 integration_rate=DEFAULT_INTEGRATION_RATE, integrator_mode=DEFAULT_INTEGRATOR_MODE,
                 noise_std=DEFAULT_NOISE_STD, fast_layer_size=DEFAULT_FAST_LAYER_SIZE,
                 fast_learning_rate=DEFAULT_FAST_LEARNING_RATE, fast_bias=DEFAULT_FAST_BIAS,
                 accumulator_rate=DEFAULT_ACCUMULATOR_RATE, accumulator_noise_std=DEFAULT_ACCUMULATOR_NOISE_STD,
                 accumulator_threshold=DEFAULT_ACCUMULATOR_THRESHOLD,
                 weight_init_scale=DEFAULT_WEIGHT_INIT_SCALE, name=DEFAULT_NAME):

        self.num_features = num_features
        self.fast_path = fast_path
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
        self.hidden_learning_rate = hidden_learning_rate
        self.integration_rate = integration_rate
        self.integrator_mode = integrator_mode
        self.noise_std = noise_std
        self.hidden_bias = hidden_bias
        self.hidden_gain = hidden_gain

        self.fast_layer_size = fast_layer_size
        self.fast_learning_rate = fast_learning_rate
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
        # Inputs
        self.color_input_layer = pnl.TransferMechanism(size=self.num_features, name='color_input')
        self.shape_input_layer = pnl.TransferMechanism(size=self.num_features, name='shape_input')

        # Task units
        self.color_task_layer = pnl.TransferMechanism(size=1, name='color_task')
        self.shape_task_layer = pnl.TransferMechanism(size=1, name='shape_task')

        # Hidden layers
        self.color_hidden_layer = pnl.TransferMechanism(size=self.hidden_layer_size,
                                                        name='color_hidden',
                                                        function=pnl.Logistic(gain=self.hidden_gain, bias=self.hidden_bias),
                                                        integrator_mode=self.integrator_mode,
                                                        integration_rate=self.integration_rate,
                                                        noise=self._generate_noise_function())
        self.shape_hidden_layer = pnl.TransferMechanism(size=self.hidden_layer_size,
                                                        name='shape_hidden',
                                                        function=pnl.Logistic(gain=self.hidden_gain, bias=self.hidden_bias),
                                                        integrator_mode=self.integrator_mode,
                                                        integration_rate=self.integration_rate,
                                                        noise=self._generate_noise_function())
        if self.fast_path:
            self.fast_shape_layer = pnl.TransferMechanism(size=self.fast_layer_size,
                                                          name='fast_shape',
                                                          function=pnl.Logistic(bias=self.fast_bias))

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
        return pnl.NormalDist(standard_dev=self.noise_std).function

    def _should_load(self, key):
        return self.loaded_weights is not None and key in self.loaded_weights

    def _generate_process(self, weight_key, in_size, out_size, in_layer, out_layer, name, learning_rate=None):
        if self._should_load(weight_key):
            weights = self.loaded_weights[weight_key]
        else:
            weights = pnl.random_matrix(in_size, out_size, 2, -1) * self.weight_init_scale

        process = pnl.Process(pathway=[in_layer, weights, out_layer],
                              name=name, learning=self.learning)
        self.processes.append(process)

        if learning_rate is None:
            learning_rate = self.hidden_learning_rate
        process.pathway[1].learning_mechanism.learning_rate = learning_rate

        return process

    def _generate_processes(self):
        self.processes = []
        self.color_hidden_process = self._generate_process(COLOR_HIDDEN_KEY,
                                                           self.num_features, self.hidden_layer_size,
                                                           self.color_input_layer, self.color_hidden_layer,
                                                           'color-hidden-proc')

        self.task_color_hidden_process = self._generate_process(COLOR_TASK_HIDDEN_KEY,
                                                                1, self.hidden_layer_size,
                                                                self.color_task_layer, self.color_hidden_layer,
                                                                'task-color-hidden-proc')

        self.shape_hidden_process = self._generate_process(SHAPE_HIDDEN_KEY,
                                                           self.num_features, self.hidden_layer_size,
                                                           self.shape_input_layer, self.shape_hidden_layer,
                                                           'shape-hidden-proc')

        self.task_shape_hidden_process = self._generate_process(SHAPE_TASK_HIDDEN_KEY,
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
                                                           'shape-output-proc')

        self.first_accumulator_process = self._generate_process(FIRST_ACCUMULATOR_DIFFERENCING_KEY,
                                                                self.num_features, self.num_features,
                                                                self.output_layer, self.first_accumulator)

        self.second_accumulator_process = self._generate_process(SECOND_ACCUMULATOR_DIFFERENCING_KEY,
                                                                self.num_features, self.num_features,
                                                                self.output_layer, self.second_accumulator)

        self.integrating_mechanisms = (self.color_hidden_layer, self.shape_hidden_layer, self.output_layer)

        if self.fast_path:
            self.shape_fast_process = self._generate_process(SHAPE_FAST_KEY,
                                                             self.hidden_layer_size, self.fast_layer_size,
                                                             self.shape_hidden_layer, self.fast_shape_layer,
                                                             'shape-fast-proc', self.fast_learning_rate)

            self.fast_output_process = self._generate_process(FAST_OUTPUT_KEY,
                                                              self.fast_layer_size, self.num_features,
                                                              self.fast_shape_layer, self.output_layer,
                                                              'fast-output-proc', self.fast_learning_rate)

    def _generate_system(self):
        # Adding learning rates to processes, as they have different ones, rather than here
        self.system = pnl.System(
            name=self.name,
            processes=self.processes,
            # learning_rate=self.learning_rate
        )

    def _switch_trial_type(self):
        if isinstance(self.system.termination_processing[pnl.TimeScale.TRIAL], pnl.AllHaveRun):
            def pass_threshold(first_mechanism, second_mechanism, threshold):
                # TODO: verify these are indeed numpy arrays
                if np.any(first_mechanism.output_states[0].value >= threshold) or \
                        np.any(second_mechanism.output_states[0].value >= threshold):
                    return True

                return False

            self._switch_trial_settings(True, self._generate_noise_function(),
                                        pnl.While(pass_threshold, self.first_accumulator,
                                                  self.second_accumulator, self.accumulator_threshold))
        else:
            self._switch_trial_settings()

    def _switch_trial_settings(self, integrator_mode=False, noise=0, termination=pnl.AllHaveRun()):
        self._switch_integrator_mode(integrator_mode)
        self._switch_noise(noise)
        self.system.termination_processing = {pnl.TimeScale.TRIAL: termination}

    def _switch_integrator_mode(self, new_mode=False):
        for mechanism in self.integrating_mechanisms:
            mechanism.integrator_mode = new_mode

    def _switch_noise(self, noise=0):
        for mechanism in self.integrating_mechanisms:
            mechanism.noise = noise

    def _repeat_tile(self, array, repeats):
        dims = np.ones(array.ndim, dtype=int)
        dims[0] = repeats
        return np.tile(array, dims)

    def train(self, inputs, tasks, targets, iterations=1, randomize=True, save_path=None,
              threshold=DEFAULT_STOPPING_THRESHOLD, report_interval=1, repeats=1):
        # TODO: REWRITE THIS COMPLETLEY TO ACCOMODATE FOR HOW THESE TRIALS NEED TO BE RAN

        mse_log = []
        times = []
        num_trials = inputs.shape[0] * repeats

        outputs = np.zeros((iterations, num_trials, 1, np.product(targets.shape[1:])))

        for iteration in range(iterations):
            if (iteration + 1) % report_interval == 0:
                print('Starting iteration {iter}'.format(iter=iteration + 1))

            times.append(time.time())

            inputs_copy = self._repeat_tile(inputs, repeats)
            task_copy = self._repeat_tile(tasks, repeats)
            target_copy = self._repeat_tile(targets, repeats)

            if randomize:
                perm = np.random.permutation(num_trials)
            else:
                perm = range(num_trials)

            input_dict = dict()
            for index, in_layer in enumerate([self.color_input_layer, self.shape_input_layer]):
                input_dict[in_layer] = inputs_copy[perm, index, :]
            input_dict[self.task_layer] = task_copy[perm, :]
            target_dict = {self.output_layer: np.squeeze(target_copy[perm])}

            output = np.array(self.system.run(inputs=input_dict, targets=target_dict)[-num_trials:])
            outputs[iteration] = output

            mse = mean_squared_error(np.ravel(target_copy[perm]), np.ravel(output))
            mse_log.append(mse)

            if (iteration + 1) % report_interval == 0:
                print('MSE after iteration {iter} is {mse}'.format(iter=iteration + 1, mse=mse))
                if save_path:
                    io.savemat(save_path, {
                        'outputs': outputs,
                        'mse': mse_log,
                        'times': times,
                        # TASK_HIDDEN_KEY: task_hidden_weights,
                        # INPUT_HIDDEN_KEY: input_hidden_weights,
                        # TASK_OUTPUT_KEY: task_output_weights,
                        # HIDDEN_OUTPUT_KEY: hidden_output_weights
                    })

            # task_hidden_weights[iteration, :, :] = self.task_hidden_process.pathway[1].function_params['matrix'][0]
            # for index in range(self.num_dimensions):
            #     input_hidden_weights[iteration, index, :, :] = self.input_hidden_processes[index].pathway[1].function_params['matrix'][0]
            #     task_output_weights[iteration, index, :, :] = self.task_output_processes[index].pathway[1].function_params['matrix'][0]
            #     hidden_output_weights[iteration, index, :, :] = self.hidden_output_processes[index].pathway[1].function_params['matrix'][0]

            if mse < threshold:
                print('MSE smaller than threshold ({threshold}), breaking'.format(threshold=threshold))
                break

        return mse_log, {
            'outputs': outputs,
            'mse': mse_log,
            'times': times,
            # TASK_HIDDEN_KEY: task_hidden_weights,
            # INPUT_HIDDEN_KEY: input_hidden_weights,
            # TASK_OUTPUT_KEY: task_output_weights,
            # HIDDEN_OUTPUT_KEY: hidden_output_weights
        }

