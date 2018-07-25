import numpy as np
import psyneulink as pnl
import multitasking_model
import single_layer_multitasking_model
from pattern_generation import generate_training_patterns
from scipy import io

# Setting up defaults
DEFAULT_TRAIN_ITERATIONS = 500
DEFAULT_NUM_REPLICATIONS = 100
DEFUALT_TASK_IDS = [0, 4, 5, 8]

DEFAULT_NUM_DIMENSIONS = 3
DEFAULT_NUM_FEATURES_PER_DIMENSION = 4

WEIGHT_FILE = r'/Users/guydavidson/Dropbox/Multitasking Experiment V2/Guy/weights/weights-3-dims-4-feats-19-Jul-2018.mat'
MATLAB_PATTERN_FILE = r'/Users/guydavidson/Dropbox/Multitasking Experiment V2/Guy/trainingSets/MIS_simulationDemo_3P4F.mat'


def main():
    # model = multitasking_model.MultitaskingModel(DEFAULT_NUM_DIMENSIONS,
    #                                              DEFAULT_NUM_FEATURES_PER_DIMENSION,
    #                                              WEIGHT_FILE)
    # model.system.show_graph(show_dimensions=pnl.ALL, show_projection_labels=pnl.ALL, show_processes=pnl.ALL)

    model = single_layer_multitasking_model.SingleLayerMultitaskingModel(DEFAULT_NUM_DIMENSIONS,
                                                                         DEFAULT_NUM_FEATURES_PER_DIMENSION,
                                                                         WEIGHT_FILE)
    # model.system.show_graph(show_dimensions=pnl.ALL, show_projection_labels=pnl.ALL, show_processes=pnl.ALL)

    # print(model.train([[[1, 0, 0, 0]], [[0, 1, 0, 0]], [[0, 0, 1, 0]]],
    #           [[1, 0, 0, 0, 0, 0, 0, 0, 0]],
    #           [[[1, 0, 0, 0]], [[0, 0, 0, 0]], [[0, 0, 0, 0]]]))

    # input_patterns, task_patterns, in_out_map, target_patterns = \
    #     generate_training_patterns(DEFAULT_NUM_DIMENSIONS, DEFAULT_NUM_FEATURES_PER_DIMENSION)
    # task_indices = np.argmax(task_patterns, 1)
    # trial_indices = np.isin(task_indices, DEFUALT_TASK_IDS)

    matlab_patterns = io.loadmat(MATLAB_PATTERN_FILE)
    input_patterns = matlab_patterns['input']
    task_patterns = matlab_patterns['tasks']
    target_patterns = matlab_patterns['train']
    task_indices = np.argmax(task_patterns, 1)
    trial_indices = np.isin(task_indices, DEFUALT_TASK_IDS)

    outputs = model.train(
        input_patterns[trial_indices], task_patterns[trial_indices], target_patterns[trial_indices], 500,
        save_path=r'/Users/guydavidson/projects/nivlab/multitasking-modeling/single-model-random-order-outputs.mat')


if __name__ == '__main__':
    main()
