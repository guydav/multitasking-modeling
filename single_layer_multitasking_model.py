import numpy as np
import psyneulink as pnl
from psyneulink.compositions.parsingautodiffcomposition import ParsingAutodiffComposition
import itertools
from sklearn.metrics import mean_squared_error
from scipy import io
import time

# Setting up default network parameters
DEFAULT_HIDDEN_PATH_SIZE = 1
DEFAULT_OUTPUT_PATH_SIZE = 1
DEFAULT_LEARNING_RATE = 0.3
DEFAULT_DECAY = 0
DEFAULT_BIAS = -2
DEFAULT_WEIGHT_INIT_SCALE = 2e-2
DEFAULT_HIDDEN_LAYER_SIZE = 200
DEFAULT_NAME = 'Single-Layer-Multitasking'

# Runtime/training parameters
DEFAULT_STOPPING_THRESHOLD = 1e-4

# Loaded weight parameters
INPUT_HIDDEN_KEY = 'weightsInputHidden'
TASK_HIDDEN_KEY = 'weightsTaskHidden'
TASK_OUTPUT_KEY = 'weightsTaskOutput'
HIDDEN_OUTPUT_KEY = 'weightsHiddenOutput'


class SingleLayerMultitaskingModel:
    def __init__(self, num_dimensions, num_features, weight_file=None, learning=pnl.LEARNING, *,
                 hidden_layer_size=DEFAULT_HIDDEN_LAYER_SIZE,
                 learning_rate=DEFAULT_LEARNING_RATE,
                 bias=DEFAULT_BIAS,
                 weight_init_scale=DEFAULT_WEIGHT_INIT_SCALE,
                 decay=DEFAULT_DECAY,
                 hidden_path_size=DEFAULT_HIDDEN_PATH_SIZE,
                 output_path_size=DEFAULT_OUTPUT_PATH_SIZE,
                 name=DEFAULT_NAME):

        self.num_dimensions = num_dimensions
        self.num_features = num_features
        if weight_file is not None:
            self.loaded_weights = io.loadmat(weight_file)
        else:
            self.loaded_weights = None
        self.learning = learning

        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate
        self.bias = bias
        self.weight_init_scale = weight_init_scale
        self.decay = decay
        self.hidden_path_size = hidden_path_size
        self.output_path_size = output_path_size
        self.name = name

        # implement equivalents of setData, configure, and constructor
        self.num_tasks = self.num_dimensions ** 2

        # Here we would initialize the layer - instead initializing the PNL model:
        self._generate_layers()
        self._initialize_weights()
        self._generate_processes()
        self._generate_system()

    def _generate_layers(self, bias=0, gain=1):
        self.task_layer = pnl.TransferMechanism(size=self.num_tasks,
                                                name='task_input')
        self.hidden_layer = pnl.TransferMechanism(size=self.hidden_layer_size,
                                                  name='hidden',
                                                  function=pnl.Logistic(bias=bias, gain=gain))
        self.hidden_bias = pnl.TransferMechanism(default_variable=np.ones((self.hidden_layer_size,)) * self.bias,
                                                 name='hidden bias')
        self.input_layer = pnl.TransferMechanism(size=self.num_dimensions * self.num_features,
                                                 name='input')
        self.output_layer = pnl.TransferMechanism(size=self.num_dimensions * self.num_features,
                                                  name='output',
                                                  function=pnl.Logistic(bias=bias, gain=gain))
        self.output_bias = pnl.TransferMechanism(
            default_variable=np.ones((self.num_dimensions * self.num_features,)) * self.bias,
            name='output-bias')

    def _should_load(self, key):
        return self.loaded_weights is not None and key in self.loaded_weights

    def _initialize_weights(self):
        if self._should_load(TASK_HIDDEN_KEY):
            self.task_hidden_weights = self.loaded_weights[TASK_HIDDEN_KEY].T
        else:
            self.task_hidden_weights = pnl.random_matrix(self.num_tasks,
                                                    self.hidden_layer_size, 2, -1) * self.weight_init_scale

        if self._should_load(INPUT_HIDDEN_KEY):
            self.input_hidden_weights = self.loaded_weights[INPUT_HIDDEN_KEY].T
        else:
            self.input_hidden_weights = pnl.random_matrix(self.num_features,
                                                     self.hidden_layer_size, 2, -1) * self.weight_init_scale

        if self._should_load(HIDDEN_OUTPUT_KEY):
            self.hidden_output_weights = self.loaded_weights[HIDDEN_OUTPUT_KEY].T
        else:
            self.hidden_output_weights = pnl.random_matrix(self.hidden_layer_size,
                                                      self.num_features, 2, -1) * self.weight_init_scale

        if self._should_load(TASK_OUTPUT_KEY):
            self.task_output_weights = self.loaded_weights[TASK_OUTPUT_KEY].T
        else:
            self.task_output_weights = pnl.random_matrix(self.num_tasks,
                                                    self.num_features, 2, -1) * self.weight_init_scale

    def _generate_processes(self):
        self.task_hidden_process = pnl.Process(pathway=[self.task_layer,
                                                        self.task_hidden_weights,
                                                        self.hidden_layer],
                                               name='task-hidden-proc',
                                               learning=pnl.ENABLED)

        self.hidden_bias_process = pnl.Process(pathway=[self.hidden_bias,
                                                        self.hidden_layer],
                                               name='hidden-bias-proc')

        self.input_hidden_process = pnl.Process(pathway=[self.input_layer,
                                                         self.input_hidden_weights,
                                                         self.hidden_layer],
                                                name='input-to-hidden-proc',
                                                learning=pnl.ENABLED)

        self.hidden_output_process = pnl.Process(pathway=[self.hidden_layer,
                                                          self.hidden_output_weights,
                                                          self.output_layer],
                                                 name='hidden-to-output-proc',
                                                 learning=pnl.ENABLED)

        self.task_output_process = pnl.Process(pathway=[self.task_layer,
                                                        self.task_output_weights,
                                                        self.output_layer],
                                               name='task-output-proc',
                                               learning=pnl.ENABLED)

        self.output_bias_process = pnl.Process(pathway=[self.output_bias,
                                                       self.output_layer],
                                               name='output-bias-proc')

    def _generate_system(self):
        self.system = pnl.System(
            name=self.name,
            processes=[self.input_hidden_process, self.task_hidden_process, self.hidden_output_process,
                       self.task_output_process, self.hidden_bias_process, self.output_bias_process],
            learning_rate=self.learning_rate
        )

    def train(self, inputs, task, target, iterations=1, randomize=True, save_path=None,
              threshold=DEFAULT_STOPPING_THRESHOLD):
        mse_log = []
        times = []

        outputs = np.zeros((iterations, inputs.shape[0], 1, np.product(inputs.shape[1:])))
        task_hidden_weights = np.zeros(
            (iterations, *self.task_hidden_process.pathway[1].matrix.shape))
        input_hidden_weights = np.zeros((iterations,
                                         *self.input_hidden_process.pathway[1].matrix.shape))
        task_output_weights = np.zeros((iterations,
                                        *self.task_output_process.pathway[1].matrix.shape))
        hidden_output_weights = np.zeros((iterations,
                                          *self.hidden_output_process.pathway[1].matrix.shape))

        for iteration in range(iterations):
            print('Starting iteration {iter}'.format(iter=iteration + 1))
            times.append(time.time())

            num_trials = inputs.shape[0]
            if randomize:
                perm = np.random.permutation(num_trials)
            else:
                perm = range(num_trials)

            input_copy = np.copy(inputs)
            task_copy = np.copy(task)
            target_copy = np.copy(target)

            input_dict = dict()
            input_dict[self.input_layer] = input_copy.reshape((inputs.shape[0], np.product(inputs.shape[1:])))[perm, :]
            input_dict[self.task_layer] = task_copy[perm, :]
            target_dict = dict()
            target_dict[self.output_layer] = target_copy.reshape((target.shape[0], np.product(target.shape[1:])))[perm, :]

            # TODO: remove this once default values properly supported
            input_dict[self.hidden_bias] = np.ones((num_trials, self.hidden_layer_size)) * self.bias
            input_dict[self.output_bias] = np.ones((num_trials, self.num_features * self.num_dimensions)) * self.bias

            if self.learning:
                output = np.array(self.system.run(inputs=input_dict, targets=target_dict)[-num_trials:])
            else:
                output = np.array(self.system.run(inputs=input_dict)[-num_trials:])

            mse = mean_squared_error(np.ravel(target[perm, :]), np.ravel(output))
            mse_log.append(mse)
            print('MSE after iteration {iter} is {mse}'.format(iter=iteration + 1, mse=mse))

            outputs[iteration, :, :] = output
            task_hidden_weights[iteration, :, :] = self.task_hidden_process.pathway[1].matrix
            input_hidden_weights[iteration, :, :] = self.input_hidden_process.pathway[1].matrix
            task_output_weights[iteration, :, :] = self.task_output_process.pathway[1].matrix
            hidden_output_weights[iteration, :, :] = self.hidden_output_process.pathway[1].matrix

            if save_path:
                io.savemat(save_path, {
                    'outputs': outputs,
                    'mse': mse_log,
                    'times': times,
                    TASK_HIDDEN_KEY: task_hidden_weights,
                    INPUT_HIDDEN_KEY: input_hidden_weights,
                    TASK_OUTPUT_KEY: task_output_weights,
                    HIDDEN_OUTPUT_KEY: hidden_output_weights
                })

            if mse < threshold:
                print('MSE smaller than threshold ({threshold}), breaking'.format(threshold=threshold))
                break

        return {
            'outputs': outputs,
            'mse': mse_log,
            'times': times,
            TASK_HIDDEN_KEY: task_hidden_weights,
            INPUT_HIDDEN_KEY: input_hidden_weights,
            TASK_OUTPUT_KEY: task_output_weights,
            HIDDEN_OUTPUT_KEY: hidden_output_weights
        }


class PyTorchMultitaskingModel(SingleLayerMultitaskingModel):
    def __init__(self, num_dimensions, num_features, weight_file=None, learning=pnl.LEARNING, *,
                 hidden_layer_size=DEFAULT_HIDDEN_LAYER_SIZE,
                 learning_rate=DEFAULT_LEARNING_RATE,
                 bias=DEFAULT_BIAS,
                 weight_init_scale=DEFAULT_WEIGHT_INIT_SCALE,
                 decay=DEFAULT_DECAY,
                 hidden_path_size=DEFAULT_HIDDEN_PATH_SIZE,
                 output_path_size=DEFAULT_OUTPUT_PATH_SIZE,
                 name=DEFAULT_NAME):

        self.composition = ParsingAutodiffComposition(param_init_from_pnl=True)
        super(PyTorchMultitaskingModel, self).__init__(num_dimensions, num_features, weight_file,
                                                       learning,
                                                       hidden_layer_size=hidden_layer_size,
                                                       learning_rate=learning_rate,
                                                       bias=bias,
                                                       weight_init_scale=weight_init_scale,
                                                       decay=decay,
                                                       hidden_path_size=hidden_path_size,
                                                       output_path_size=output_path_size,
                                                       name=name)

    def _generate_layers(self, bias=0, gain=1):
        # PNL subtracts bias rather than adding it, so we multiply by -1
        super(PyTorchMultitaskingModel, self)._generate_layers(bias=-1 * self.bias)

        # No bias layers for now
        for layer in (self.input_layer, self.hidden_layer, self.task_layer, self.output_layer):
            self.composition.add_c_node(layer)

    def _generate_processes(self):
        # task => hidden
        self.composition.add_projection(sender=self.task_layer, projection=self.task_hidden_weights,
                                        receiver=self.hidden_layer)

        # input => hidden
        self.composition.add_projection(sender=self.input_layer, projection=self.input_hidden_weights,
                                        receiver=self.hidden_layer)

        # hidden => output
        self.composition.add_projection(sender=self.hidden_layer, projection=self.hidden_output_weights,
                                        receiver=self.output_layer)

        # task => output
        self.composition.add_projection(sender=self.task_layer, projection=self.task_output_weights,
                                        receiver=self.output_layer)

    def _generate_system(self):
        pass

    def train(self, inputs, task, target, iterations=1, randomize=True, save_path=None,
              threshold=DEFAULT_STOPPING_THRESHOLD):

        inputs = inputs.reshape((inputs.shape[0], np.product(inputs.shape[1:])))
        target = target.reshape((target.shape[0], np.product(target.shape[1:])))

        input_dict = dict()
        input_dict[self.input_layer] = inputs
        input_dict[self.task_layer] = task

        target_dict = dict()
        target_dict[self.output_layer] = target

        if self.learning:
            last_outputs = self.composition.run(inputs=input_dict, targets=target_dict, epochs=iterations,
                                                randomize=randomize, learning_rate=self.learning_rate)
        else:
            last_outputs = self.composition.run(inputs=input_dict)

        return {
            'last_outputs': last_outputs,
            'mse': self.composition.losses,
        }
