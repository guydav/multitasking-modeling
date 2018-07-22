import numpy as np
import psyneulink as pnl
import itertools
from sklearn.metrics import mean_squared_error
from scipy import io


# Setting up default network parameters
DEFAULT_HIDDEN_PATH_SIZE = 1
DEFAULT_OUTPUT_PATH_SIZE = 1
DEFAULT_LEARNING_RATE = 0.3
DEFAULT_DECAY = 0
DEFAULT_BIAS = -2
DEFAULT_WEIGHT_INIT_SCALE = 2e-2
DEFAULT_HIDDEN_LAYER_SIZE = 200

# Runtime/training parameters
DEFAULT_STOPPING_THRESHOLD = 1e-4

# Loaded weight parameters
INPUT_HIDDEN_KEY = 'weightsInputHidden'
TASK_HIDDEN_KEY = 'weightsTaskHidden'
TASK_OUTPUT_KEY = 'weightsTaskOutput'
HIDDEN_OUTPUT_KEY = 'weightsHiddenOutput'


class MultitaskingModel:
    def __init__(self, num_dimensions, num_features, weight_file=None, learning=pnl.ENABLED, *,
                 hidden_layer_size=DEFAULT_HIDDEN_LAYER_SIZE,
                 learning_rate=DEFAULT_LEARNING_RATE,
                 bias=DEFAULT_BIAS,
                 weight_init_scale=DEFAULT_WEIGHT_INIT_SCALE,
                 decay=DEFAULT_DECAY,
                 hidden_path_size=DEFAULT_HIDDEN_PATH_SIZE,
                 output_path_size=DEFAULT_OUTPUT_PATH_SIZE):

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

        # implement equivalents of setData, configure, and constructor
        self.num_tasks = self.num_dimensions ** 2

        # Here we would initialize the layer - instead initializing the PNL model:
        self.task_layer = pnl.TransferMechanism(size=self.num_tasks,
                                                name='task_input')
        self.hidden_layer = pnl.TransferMechanism(size=self.hidden_layer_size,
                                                  name='hidden',
                                                  function=pnl.Logistic)
        self.hidden_bias = pnl.TransferMechanism(default_variable=np.ones((self.hidden_layer_size,)) * self.bias,
                                                 name='hidden bias')
        self.input_layers = self._generate_io_layers('input')
        self.output_layers = self._generate_io_layers('output', func=pnl.Logistic)
        self._generate_output_bias_layers()
        self.input_output_processes = []
        # self._generate_processes_product()
        self._generate_processes()
        self._generate_system()

    def _generate_io_layers(self, name, func=pnl.Linear):
        return [pnl.TransferMechanism(size=self.num_features,
                                      name='{n}-{i}'.format(n=name, i=i),
                                      function=func)
                for i in range(self.num_dimensions)]

    def _generate_output_bias_layers(self):
        self.output_biases = [
            pnl.TransferMechanism(default_variable=np.ones((self.num_features,)) * self.bias,
                                  name='output-bias-{i}'.format(i=i))
            for i in range(self.num_dimensions)]

    # def _generate_processes_product(self):
    #     for (input_index, output_index) in itertools.product(range(self.num_dimensions),
    #                                                          range(self.num_dimensions)):
    #         proc = pnl.Process(pathway=[self.input_layers[input_index],
    #                                     pnl.random_matrix(self.num_features, self.hidden_layer_size, 2, -1) * self.weight_init_scale,
    #                                     self.hidden_layer,
    #                                     pnl.random_matrix(self.hidden_layer_size, self.num_features, 2, -1) * self.weight_init_scale,
    #                                     self.output_layers[output_index]],
    #                            name='input-{i}-output-{o}-proc'.format(i=input_index, o=output_index),
    #                            learning=self.learning)
    #
    #         self.input_output_processes.append(proc)

    def _should_load(self, key):
        return self.loaded_weights is not None and key in self.loaded_weights

    def _generate_processes(self):
        if self._should_load(TASK_HIDDEN_KEY):
            task_hidden_weights = self.loaded_weights[TASK_HIDDEN_KEY].T
        else:
            task_hidden_weights = pnl.random_matrix(self.num_tasks,
                                                    self.hidden_layer_size, 2, -1) * self.weight_init_scale
        self.task_hidden_process = pnl.Process(pathway=[self.task_layer,
                                                        task_hidden_weights,
                                                        self.hidden_layer],
                                               name='task-hidden-proc',
                                               learning=pnl.ENABLED)

        self.hidden_bias_process = pnl.Process(pathway=[self.hidden_bias,
                                                        self.hidden_layer],
                                               name='hidden-bias-proc')

        self.input_hidden_processes = []
        self.hidden_output_processes = []
        self.task_output_processes = []
        self.output_bias_processes = []

        for index in range(self.num_dimensions):
            if self._should_load(INPUT_HIDDEN_KEY):
                input_hidden_weights = self.loaded_weights[INPUT_HIDDEN_KEY][:, index * self.num_features:
                                                                                (index + 1) * self.num_features].T
            else:
                input_hidden_weights = pnl.random_matrix(self.num_features,
                                                         self.hidden_layer_size, 2, -1) * self.weight_init_scale

            self.input_hidden_processes.append(pnl.Process(pathway=[self.input_layers[index],
                                                                    input_hidden_weights,
                                                                    self.hidden_layer],
                                                           name='input-{i}-to-hidden-proc'.format(i=index),
                                                           learning=pnl.ENABLED))

            if self._should_load(HIDDEN_OUTPUT_KEY):
                hidden_output_weights = self.loaded_weights[HIDDEN_OUTPUT_KEY][index * self.num_features:
                                                                               (index + 1) * self.num_features, :].T
            else:
                hidden_output_weights = pnl.random_matrix(self.hidden_layer_size,
                                                          self.num_features, 2, -1) * self.weight_init_scale

            self.hidden_output_processes.append(pnl.Process(pathway=[self.hidden_layer,
                                                                     hidden_output_weights,
                                                                     self.output_layers[index]],
                                                            name='hidden-to-output-{o}-proc'.format(o=index),
                                                            learning=pnl.ENABLED))

            if self._should_load(TASK_OUTPUT_KEY):
                task_output_weights = self.loaded_weights[TASK_OUTPUT_KEY][index * self.num_features:
                                                                               (index + 1) * self.num_features, :].T
            else:
                task_output_weights = pnl.random_matrix(self.num_tasks,
                                                        self.num_features, 2, -1) * self.weight_init_scale

            self.task_output_processes.append(
                pnl.Process(pathway=[self.task_layer,
                                     task_output_weights,
                                     self.output_layers[index]],
                            name='task-output-proc-{o}'.format(o=index),
                            learning=pnl.ENABLED))

            self.output_bias_processes.append(
                pnl.Process(pathway=[self.output_biases[index],
                                     self.output_layers[index]],
                            name='output-bias-proc-{o}'.format(o=index)))

    def _generate_system(self):
        self.system = pnl.System(processes=self.input_hidden_processes +
                                           self.hidden_output_processes +
                                           self.task_output_processes +
                                           self.input_output_processes +
                                           self.output_bias_processes +
                                           [self.task_hidden_process, self.hidden_bias_process],
                                 learning_rate=self.learning_rate
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
                print('MSE smaller than threshold ({threshold}, breaking'.format(threshold=threshold))
                break

        return mse_log, {
            'outputs': outputs,
            'mse': mse_log,
            TASK_HIDDEN_KEY: task_hidden_weights,
            INPUT_HIDDEN_KEY: input_hidden_weights,
            TASK_OUTPUT_KEY: task_output_weights,
            HIDDEN_OUTPUT_KEY: hidden_output_weights
        }

