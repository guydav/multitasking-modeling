import numpy as np
import psyneulink as pnl
import itertools
from sklearn.metrics import mean_squared_error
from scipy import io


# Setting up default network parameters
#  TODO: find the correct values for all of these
DEFAULT_HIDDEN_LAYER_SIZE = 100
DEFAULT_LEARNING_RATE = 0.3

DEFAULT_FAST_LEARNING_RATE = 1.0
DEFAULT_FAST_LAYER_SIZE = 50

DEFAULT_GAIN = 1
DEFAULT_BIAS = -2
DEFAULT_WEIGHT_INIT_SCALE = 2e-2

DEFAULT_NAME = 'Shape-Naming'

# Runtime/training parameters
DEFAULT_STOPPING_THRESHOLD = 1e-4

# Loaded weight parameters
COLOR_HIDDEN_KEY = 'weightsColorHidden'
TASK_COLOR_HIDDEN_KEY = 'weightsTaskColorHidden'

SHAPE_HIDDEN_KEY = 'weightsShapeHidden'
TASK_SHAPE_HIDDEN_KEY = 'weightsTaskShapeHidden'

COLOR_OUTPUT_KEY = 'weightsColorOutput'
SHAPE_OUTPUT_KEY = 'weightsShapeOutput'

SHAPE_FAST_KEY = 'weightsShapeFast'
FAST_OUTPUT_KEY = 'weightsFastOutput'

NUM_TASKS = 2


class ShapeNamingModel:
    def __init__(self, num_features, fast_path=True, weight_file=None, learning=pnl.LEARNING, *,
                 hidden_layer_size=DEFAULT_HIDDEN_LAYER_SIZE,
                 learning_rate=DEFAULT_LEARNING_RATE,
                 fast_layer_size=DEFAULT_FAST_LAYER_SIZE,
                 fast_learning_rate=DEFAULT_FAST_LEARNING_RATE,
                 bias=DEFAULT_BIAS,
                 gain=DEFAULT_GAIN,
                 weight_init_scale=DEFAULT_WEIGHT_INIT_SCALE,
                 name=DEFAULT_NAME):

        self.num_features = num_features
        self.fast_path = fast_path
        if weight_file is not None:
            self.loaded_weights = io.loadmat(weight_file)
        else:
            self.loaded_weights = None
        self.learning = learning

        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate
        self.fast_layer_size = fast_layer_size
        self.fast_learning_rate = fast_learning_rate
        self.bias = bias
        self.gain = gain
        self.weight_init_scale = weight_init_scale
        self.name = name

        self._generate_layers()
        self._generate_processes()
        self._generate_system()

    def _generate_layers(self):
        # Inputs
        self.color_input_layer = pnl.TransferMechanism(size=self.num_features, name='color_input')
        self.shape_input_layer = pnl.TransferMechanism(size=self.num_features, name='shape_input')
        self.task_layer = pnl.TransferMechanism(size=NUM_TASKS, name='task_input')

        # Hidden layers
        self.color_hidden_layer = pnl.TransferMechanism(size=self.hidden_layer_size,
                                                        name='color_hidden',
                                                        function=pnl.Logistic(gain=self.gain, bias=self.bias))
        self.shape_hidden_layer = pnl.TransferMechanism(size=self.hidden_layer_size,
                                                        name='shape_hidden',
                                                        function=pnl.Logistic(gain=self.gain, bias=self.bias))
        if self.fast_path:
            self.fast_shape_layer = pnl.TransferMechanism(size=self.fast_layer_size,
                                                          name='fast_shape',
                                                          function=pnl.Logistic(gain=self.gain, bias=self.bias))

        # Output layers
        self.output_layer = pnl.TransferMechanism(size=self.num_features,
                                                  name='output',
                                                  function=pnl.Logistic(gain=self.gain, bias=self.bias))

    def _should_load(self, key):
        return self.loaded_weights is not None and key in self.loaded_weights

    def _generate_process(self, weight_key, in_size, out_size, in_layer, out_layer, name, learning_rate=None):
        if self._should_load(weight_key):
            weights = self.loaded_weights[weight_key].T
        else:
            weights = pnl.random_matrix(in_size, out_size, 2, -1) * self.weight_init_scale

        process = pnl.Process(pathway=[in_layer, weights, out_layer],
                              name=name, learning=self.learning)
        self.processes.append(process)

        if learning_rate is None:
            learning_rate = self.learning_rate
        process.pathway[1].learning_mechanism.learning_rate = learning_rate

        return process

    def _generate_processes(self):
        self.processes = []
        self.color_hidden_process = self._generate_process(COLOR_HIDDEN_KEY,
                                                           self.num_features, self.hidden_layer_size,
                                                           self.color_input_layer, self.color_hidden_layer,
                                                           'color-hidden-proc')

        self.task_color_hidden_process = self._generate_process(TASK_COLOR_HIDDEN_KEY,
                                                                NUM_TASKS, self.hidden_layer_size,
                                                                self.task_layer,  self.color_hidden_layer,
                                                                'task-color-hidden-proc')

        self.shape_hidden_process = self._generate_process(SHAPE_HIDDEN_KEY,
                                                           self.num_features, self.hidden_layer_size,
                                                           self.shape_input_layer, self.shape_hidden_layer,
                                                           'shape-hidden-proc')

        self.task_shape_hidden_process = self._generate_process(TASK_SHAPE_HIDDEN_KEY,
                                                                NUM_TASKS, self.hidden_layer_size,
                                                                self.task_layer, self.shape_hidden_layer,
                                                                'task-shape-hidden-proc')

        self.color_output_process = self._generate_process(COLOR_OUTPUT_KEY,
                                                           self.hidden_layer_size, self.num_features,
                                                           self.color_hidden_layer, self.output_layer,
                                                           'color-output-proc')

        self.shape_output_process = self._generate_process(SHAPE_OUTPUT_KEY,
                                                           self.hidden_layer_size, self.num_features,
                                                           self.shape_hidden_layer, self.output_layer,
                                                           'shape-output-proc')

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

    def train(self, inputs, task, target, iterations=1, randomize=True, save_path=None, threshold=DEFAULT_STOPPING_THRESHOLD):
        mse_log = []

        outputs = np.zeros((iterations, *inputs.shape))
        task_hidden_weights = np.zeros((iterations, *self.task_hidden_process.pathway[1].function_params['matrix'][0].shape))
        input_hidden_weights = np.zeros((iterations, self.num_dimensions,
                                         *self.input_hidden_processes[0].pathway[1].function_params['matrix'][0].shape))
        task_output_weights = np.zeros((iterations, self.num_dimensions,
                                        *self.task_output_processes[0].pathway[1].function_params['matrix'][0].shape))
        hidden_output_weights = np.zeros((iterations, self.num_dimensions,
                                          *self.hidden_output_processes[0].pathway[1].function_params['matrix'][0].shape))

        for iteration in range(iterations):
            print('Starting iteration {iter}'.format(iter=iteration + 1))
            num_trials = inputs.shape[0]
            if randomize:
                perm = np.random.permutation(num_trials)
            else:
                perm = range(num_trials)

            input_copy = np.copy(inputs)
            task_copy = np.copy(task)
            target_copy = np.copy(target)

            input_dict = {self.input_layers[i]: input_copy[perm, i, :] for i in range(self.num_dimensions)}
            input_dict[self.task_layer] = task_copy[perm, :]
            target_dict = {self.output_layers[i]: target_copy[perm, i, :] for i in range(self.num_dimensions)}

            # TODO: remove this once default values properly supported
            input_dict[self.hidden_bias] = np.ones((num_trials, self.hidden_layer_size)) * self.bias
            input_dict.update({bias: np.ones((num_trials, self.num_features)) * self.bias for bias in self.output_biases})

            output = np.array(self.system.run(inputs=input_dict, targets=target_dict)[-num_trials:])
            mse = mean_squared_error(np.ravel(target), np.ravel(output))
            mse_log.append(mse)
            print('MSE after iteration {iter} is {mse}'.format(iter=iteration + 1, mse=mse))

            outputs[iteration, :, :, :] = output
            task_hidden_weights[iteration, :, :] = self.task_hidden_process.pathway[1].function_params['matrix'][0]
            for index in range(self.num_dimensions):
                input_hidden_weights[iteration, index, :, :] = self.input_hidden_processes[index].pathway[1].function_params['matrix'][0]
                task_output_weights[iteration, index, :, :] = self.task_output_processes[index].pathway[1].function_params['matrix'][0]
                hidden_output_weights[iteration, index, :, :] = self.hidden_output_processes[index].pathway[1].function_params['matrix'][0]

            if save_path:
                io.savemat(save_path, {
                    'outputs': outputs,
                    TASK_HIDDEN_KEY: task_hidden_weights,
                    INPUT_HIDDEN_KEY: input_hidden_weights,
                    TASK_OUTPUT_KEY: task_output_weights,
                    HIDDEN_OUTPUT_KEY: hidden_output_weights
                })

            if mse < threshold:
                print('MSE smaller than threshold ({threshold}), breaking'.format(threshold=threshold))
                break

        return mse_log, {
            'outputs': outputs,
            'mse': mse_log,
            TASK_HIDDEN_KEY: task_hidden_weights,
            INPUT_HIDDEN_KEY: input_hidden_weights,
            TASK_OUTPUT_KEY: task_output_weights,
            HIDDEN_OUTPUT_KEY: hidden_output_weights
        }

