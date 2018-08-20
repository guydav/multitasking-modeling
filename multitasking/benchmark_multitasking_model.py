import numpy as np
from multitasking import single_layer_multitasking_model
from scipy import io
import time


# Setting up defaults
TRAIN_ITERATIONS = 500
TRAIN_REPLICATIONS = 10
DEFUALT_TASK_IDS = [0, 4, 5, 8]

DEFAULT_NUM_DIMENSIONS = 3
DEFAULT_NUM_FEATURES_PER_DIMENSION = 4

WEIGHT_FILE = r'/Users/guydavidson/Dropbox/Multitasking Experiment V2/Guy/weights/weights-3-dims-4-feats-19-Jul-2018.mat'
MATLAB_PATTERN_FILE = r'/Users/guydavidson/Dropbox/Multitasking Experiment V2/Guy/trainingSets/MIS_simulationDemo_3P4F.mat'

FOLDER = r'/Users/guydavidson/projects/nivlab/multitasking-modeling/multitasking-results'


def main():
    model = single_layer_multitasking_model.PyTorchSingleLayerMultitaskingModel(DEFAULT_NUM_DIMENSIONS,
                                                                                DEFAULT_NUM_FEATURES_PER_DIMENSION,
                                                                                WEIGHT_FILE)
    # model = single_layer_multitasking_model.SingleLayerMultitaskingModel(DEFAULT_NUM_DIMENSIONS,
    #                                                                      DEFAULT_NUM_FEATURES_PER_DIMENSION,
    #                                                                      WEIGHT_FILE)

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

    times = []

    for i in range(TRAIN_REPLICATIONS):
        start = time.time()
        outputs = model.train(
            input_patterns[trial_indices], task_patterns[trial_indices], target_patterns[trial_indices],
            TRAIN_ITERATIONS, False)
        end = time.time()
        times.append(end - start)
    # print('500 iterations took {t}'.format(t=end - start))
    print('After {r} replications of {i} iteration(s), tookk a min of {min:.4f} and a mean of {mean:.4f}'.format(
        r=TRAIN_REPLICATIONS, i=TRAIN_ITERATIONS, min=min(times), mean=sum(times) / len(times)
    ))
    # io.savemat(os.path.join(FOLDER, 'pytorch-matlab-patterns-no-shuffle.mat'), outputs)


if __name__ == '__main__':
    main()
