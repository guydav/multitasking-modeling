import numpy as np
import psyneulink as pnl
import multitasking_nn_model
from pattern_generation import generate_training_patterns

# Setting up defaults
DEFAULT_TRAIN_ITERATIONS = 500
DEFAULT_NUM_REPLICATIONS = 100
DEFUALT_TASK_IDS = [0, 4, 5, 8]

DEFAULT_NUM_DIMENSIONS = 3
DEFAULT_NUM_FEATURES_PER_DIMENSION = 4


def load_or_create_patterns(num_dimensions=DEFAULT_NUM_DIMENSIONS,
                            num_features=DEFAULT_NUM_FEATURES_PER_DIMENSION):
    # TODO: get this code from Sebastian
    pass


def main():
    model = multitasking_nn_model.MultitaskingModel(DEFAULT_NUM_DIMENSIONS, DEFAULT_NUM_FEATURES_PER_DIMENSION)
    # model.system.show_graph(show_dimensions=pnl.ALL, show_projection_labels=pnl.ALL, show_processes=pnl.ALL)

    # print(model.train([[[1, 0, 0, 0]], [[0, 1, 0, 0]], [[0, 0, 1, 0]]],
    #           [[1, 0, 0, 0, 0, 0, 0, 0, 0]],
    #           [[[1, 0, 0, 0]], [[0, 0, 0, 0]], [[0, 0, 0, 0]]]))

    input_patterns, task_patterns, in_out_map, target_patterns = generate_training_patterns(DEFAULT_NUM_DIMENSIONS,
                                                                                            DEFAULT_NUM_FEATURES_PER_DIMENSION)
    task_indices = np.argmax(task_patterns, 1)
    trial_indices = np.isin(task_indices, DEFUALT_TASK_IDS)

    print(model.train(input_patterns[trial_indices], task_patterns[trial_indices], target_patterns[trial_indices], 10))


if __name__ == '__main__':
    main()
