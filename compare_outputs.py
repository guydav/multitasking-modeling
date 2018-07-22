from scipy import io
import numpy as np


PYTHON_OUTPUT = r'/Users/guydavidson/projects/nivlab/multitasking-modeling/model-outputs.mat'
PYTHON_SINGLE_LAYER_OUTPUT = r'/Users/guydavidson/projects/nivlab/multitasking-modeling/single-layer-model-outputs.mat'
PYTHON_SINGLE_LAYER_NO_LEARNING_OUTPUT = r'/Users/guydavidson/projects/nivlab/multitasking-modeling/single-layer-no-learning-model-outputs.mat'
MATLAB_OUTPUT = r'/Users/guydavidson/Dropbox/Multitasking Experiment V2/Guy/outputs/outputs-3-dims-4-feats-5-iters-20-Jul-2018.mat'

INPUT_HIDDEN_KEY = 'weightsInputHidden'
TASK_HIDDEN_KEY = 'weightsTaskHidden'
TASK_OUTPUT_KEY = 'weightsTaskOutput'
HIDDEN_OUTPUT_KEY = 'weightsHiddenOutput'
OUTPUT_KEY = 'outputs'

if __name__ == '__main__':
    # python_output = io.loadmat(PYTHON_OUTPUT)
    python_output = io.loadmat(PYTHON_SINGLE_LAYER_OUTPUT)
    # python_output = io.loadmat(PYTHON_SINGLE_LAYER_NO_LEARNING_OUTPUT)
    matlab_output = io.loadmat(MATLAB_OUTPUT)

    py_out = np.squeeze(python_output[OUTPUT_KEY])
    mat_out = matlab_output[OUTPUT_KEY]
    print(py_out.shape, mat_out.shape)
    print(py_out[0, 0, :])
    print(mat_out[0, 0, :])
    print(np.allclose(py_out[0, 0, :], mat_out[0, 0, :]))
    print(matlab_output['mse'])
    print(python_output['mse'])

