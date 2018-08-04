import numpy as np
import psyneulink as pnl
from psyneulink.compositions.parsingautodiffcomposition import ParsingAutodiffComposition
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
DEFAULT_NAME = 'Multitasking'

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

    def _generate_layers(self, bias=None, gain=1):
        if not bias:
            bias = -1 * self.bias

        self.task_layer = pnl.TransferMechanism(size=self.num_tasks,
                                                name='task_input')
        self.hidden_layer = pnl.TransferMechanism(size=self.hidden_layer_size,
                                                  name='hidden',
                                                  function=pnl.Logistic(bias=bias, gain=gain))
        # self.hidden_bias = pnl.TransferMechanism(default_variable=np.ones((self.hidden_layer_size,)) * self.bias,
        #                                          name='hidden bias')
        self.input_layers = self._generate_io_layers('input')
        self.output_layers = self._generate_io_layers('output', func=pnl.Logistic(bias=bias, gain=gain))
        # self._generate_output_bias_layers()

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

    def _should_load(self, key):
        return self.loaded_weights is not None and key in self.loaded_weights

    def _generate_processes(self):
        self.input_output_processes = []

        self.task_hidden_process = pnl.Process(pathway=[self.task_layer,
                                                        self.task_hidden_weights,
                                                        self.hidden_layer],
                                               name='task-hidden-proc',
                                               learning=pnl.ENABLED)

        # self.hidden_bias_process = pnl.Process(pathway=[self.hidden_bias,
        #                                                 self.hidden_layer],
        #                                        name='hidden-bias-proc')

        self.input_hidden_processes = []
        self.hidden_output_processes = []
        self.task_output_processes = []
        # self.output_bias_processes = []

        for index in range(self.num_dimensions):
            input_hidden_weights = self.input_hidden_weights[:, index * self.num_features:
                                                                (index + 1) * self.num_features].T
            self.input_hidden_processes.append(pnl.Process(pathway=[self.input_layers[index],
                                                                    input_hidden_weights,
                                                                    self.hidden_layer],
                                                           name='input-{i}-to-hidden-proc'.format(i=index),
                                                           learning=pnl.ENABLED))

            hidden_output_weights = self.hidden_output_weights[index * self.num_features:
                                                               (index + 1) * self.num_features, :].T
            self.hidden_output_processes.append(pnl.Process(pathway=[self.hidden_layer,
                                                                     hidden_output_weights,
                                                                     self.output_layers[index]],
                                                            name='hidden-to-output-{o}-proc'.format(o=index),
                                                            learning=pnl.ENABLED))

            task_output_weights = self.task_output_weights[index * self.num_features:
                                                           (index + 1) * self.num_features, :].T
            self.task_output_processes.append(
                pnl.Process(pathway=[self.task_layer,
                                     task_output_weights,
                                     self.output_layers[index]],
                            name='task-output-proc-{o}'.format(o=index),
                            learning=pnl.ENABLED))

            # self.output_bias_processes.append(
            #     pnl.Process(pathway=[self.output_biases[index],
            #                          self.output_layers[index]],
            #                 name='output-bias-proc-{o}'.format(o=index)))

    def _generate_system(self):
        self.system = pnl.System(name=self.name,
                                 processes=self.input_hidden_processes +
                                           self.hidden_output_processes +
                                           self.task_output_processes +
                                           self.input_output_processes +
                                           # self.output_bias_processes +
                                           [self.task_hidden_process],  #, self.hidden_bias_process],
                                 learning_rate=self.learning_rate
                                 )

    def train(self, inputs, task, target, iterations=1, randomize=True, save_path=None,
              threshold=DEFAULT_STOPPING_THRESHOLD):
        mse_log = []
        times = []

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
            times.append(time.time())

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
            # input_dict[self.hidden_bias] = np.ones((num_trials, self.hidden_layer_size)) * self.bias
            # input_dict.update({bias: np.ones((num_trials, self.num_features)) * self.bias for bias in self.output_biases})

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
                    'mse': mse_log,
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


class PyTorchMultitaskingModel(MultitaskingModel):
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

        for layer in self.input_layers + self.output_layers + [self.task_layer, self.hidden_layer]:
            self.composition.add_c_node(layer)

    def _generate_processes(self):
        # task => hidden
        self.composition.add_projection(sender=self.task_layer, projection=self.task_hidden_weights,
                                        receiver=self.hidden_layer)

        for index in range(self.num_dimensions):
            input_hidden_weights = self.input_hidden_weights[index * self.num_features:
                                                             (index + 1) * self.num_features, :]
            # input => hidden
            self.composition.add_projection(sender=self.input_layers[index],
                                            projection=input_hidden_weights,
                                            receiver=self.hidden_layer)

            hidden_output_weights = self.hidden_output_weights[:, index * self.num_features:
                                                                  (index + 1) * self.num_features]
            # hidden => output
            self.composition.add_projection(sender=self.hidden_layer,
                                            projection=hidden_output_weights,
                                            receiver=self.output_layers[index])

            task_output_weights = self.task_output_weights[:, index * self.num_features:
                                                              (index + 1) * self.num_features]
            # task => output
            self.composition.add_projection(sender=self.task_layer,
                                            projection=task_output_weights,
                                            receiver=self.output_layers[index])

    def _generate_system(self):
        pass

    def train(self, inputs, task, target, iterations=1, randomize=True, save_path=None,
              threshold=DEFAULT_STOPPING_THRESHOLD):
        input_dict = {self.input_layers[index]: inputs[:, index * self.num_features:
                                                          (index + 1) * self.num_features]
                      for index in range(self.num_dimensions)}
        input_dict[self.task_layer] = task
        target_dict = {self.output_layers[index]: target[:, index * self.num_features:
                                                            (index + 1) * self.num_features]
                       for index in range(self.num_dimensions)}

        if self.learning:
            last_outputs = self.composition.run(inputs=input_dict, targets=target_dict, epochs=iterations,
                                                randomize=randomize, learning_rate=self.learning_rate,
                                                optimizer='sgd')
        else:
            last_outputs = self.composition.run(inputs=input_dict)

        return {
            'last_outputs': last_outputs,
            'mse': self.composition.losses,
        }

