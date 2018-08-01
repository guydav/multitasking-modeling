import numpy as np
import psyneulink as pnl
import shape_naming_model
from pattern_generation import generate_training_patterns
import os

# Setting up defaults
DEFAULT_TRAIN_ITERATIONS = 500
DEFAULT_NUM_FEATURES = 2
NUM_INPUT_DIMENSIONS = 2
NUM_OUTPUT_DIMENSIONS = 1

FOLDER = r'/Users/guydavidson/projects/nivlab/multitasking-modeling/shape-naming-results'


def main():
    input_patterns, task_patterns, in_out_map, target_patterns = \
        generate_training_patterns(NUM_INPUT_DIMENSIONS, DEFAULT_NUM_FEATURES, NUM_OUTPUT_DIMENSIONS)

    for hidden_size in range(2, 6):
        print('Training model with {hs} hidden units per layer...'.format(hs=hidden_size))
        model = shape_naming_model.ShapeNamingModel(DEFAULT_NUM_FEATURES, fast_path=False, hidden_layer_size=hidden_size,
                                                    learning_rate=0.3, bias=-1)

        # model.system.show_graph(show_dimensions=pnl.ALL, show_projection_labels=pnl.ALL,
        #                         show_processes=pnl.ALL)

        output_log = model.train(
            input_patterns, task_patterns, target_patterns, 500,
            False, os.path.join(FOLDER, 'shape-naming-no-fast-adj-bias-{hs}-hidden-units.mat'.format(hs=hidden_size)),
            report_interval=5, repeats=10)


if __name__ == '__main__':
    main()
