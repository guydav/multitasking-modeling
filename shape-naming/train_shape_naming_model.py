import numpy as np
import shape_naming_model
import psyneulink as pnl
import os


# Setting up defaults
DEFAULT_TRAIN_ITERATIONS = 500
DEFAULT_NUM_FEATURES = 2
NUM_INPUT_DIMENSIONS = 2
NUM_OUTPUT_DIMENSIONS = 1

FOLDER = r'/Users/guydavidson/projects/nivlab/multitasking-modeling/shape-naming-results'


def main():
    model = shape_naming_model.ShapeNamingModel(DEFAULT_NUM_FEATURES) #, indirect_path=False)
    model.system.show_graph(show_dimensions=pnl.ALL, show_projection_labels=pnl.ALL, show_processes=pnl.ALL)
    print(model.system.scheduler_processing.consideration_queue)
    out = model.train()
    print(out)

    test_out = model.test()
    print(test_out)


if __name__ == '__main__':
    main()
