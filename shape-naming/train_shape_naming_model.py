import numpy as np
import shape_naming_model
import psyneulink as pnl
import os


FOLDER = r'/Users/guydavidson/projects/nivlab/multitasking-modeling/shape-naming-results'


def main():
    # model = shape_naming_model.ShapeNamingModel() #, indirect_path=False)
    # model.train()
    # df = model.last_run_to_dataframe()
    # print(df)

    # model.system.show_graph(show_dimensions=pnl.ALL, show_projection_labels=pnl.ALL, show_processes=pnl.ALL)
    # print(model.system.scheduler_processing.consideration_queue)
    # out = model.train()
    # print(out)
    #
    # test_out = model.test()
    # print(test_out)

    hebbian_model = shape_naming_model.HebbianShapeNamingModel(
        direct_learning_rate=0.1, noise_std=0.01, integration_rate=0.2, accumulator_noise_std=0.01,
        indirect_learning_rate=0.1, indirect_bias=2, indirect_layer_size=6)

    train_out = hebbian_model.train()
    test_out = hebbian_model.test()



if __name__ == '__main__':
    main()
