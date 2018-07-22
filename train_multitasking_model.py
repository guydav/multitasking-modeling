import numpy as np
import psyneulink as pnl
import multitasking_model
import single_layer_multitasking_model
from pattern_generation import generate_training_patterns

# Setting up defaults
DEFAULT_TRAIN_ITERATIONS = 500
DEFAULT_NUM_REPLICATIONS = 100
DEFUALT_TASK_IDS = [0, 4, 5, 8]

DEFAULT_NUM_DIMENSIONS = 3
DEFAULT_NUM_FEATURES_PER_DIMENSION = 4

WEIGHT_FILE = r'/Users/guydavidson/Dropbox/Multitasking Experiment V2/Guy/weights/weights-3-dims-4-feats-19-Jul-2018.mat'


def main():
    model = multitasking_model.MultitaskingModel(DEFAULT_NUM_DIMENSIONS,
                                                    DEFAULT_NUM_FEATURES_PER_DIMENSION,
                                                    WEIGHT_FILE)
    # model.system.show_graph(show_dimensions=pnl.ALL, show_projection_labels=pnl.ALL, show_processes=pnl.ALL)

    # model = single_layer_multitasking_model.SingleLayerMultitaskingModel(DEFAULT_NUM_DIMENSIONS,
    #                                                                      DEFAULT_NUM_FEATURES_PER_DIMENSION,
    #                                                                      WEIGHT_FILE)
    # model.system.show_graph(show_dimensions=pnl.ALL, show_projection_labels=pnl.ALL, show_processes=pnl.ALL)

    # print(model.train([[[1, 0, 0, 0]], [[0, 1, 0, 0]], [[0, 0, 1, 0]]],
    #           [[1, 0, 0, 0, 0, 0, 0, 0, 0]],
    #           [[[1, 0, 0, 0]], [[0, 0, 0, 0]], [[0, 0, 0, 0]]]))

    input_patterns, task_patterns, in_out_map, target_patterns = \
        generate_training_patterns(DEFAULT_NUM_DIMENSIONS, DEFAULT_NUM_FEATURES_PER_DIMENSION)
    task_indices = np.argmax(task_patterns, 1)
    trial_indices = np.isin(task_indices, DEFUALT_TASK_IDS)
    # trial_indices = [0]

    mse_log, output_log = model.train(
        input_patterns[trial_indices], task_patterns[trial_indices], target_patterns[trial_indices], 500,
        False, r'/Users/guydavidson/projects/nivlab/multitasking-modeling/model-outputs.mat')


if __name__ == '__main__':
    main()
