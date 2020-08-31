import os
import pickle
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/'
DATA_DIR = ROOT_DIR + 'data/'
MODEL_SAVE_DIR = ROOT_DIR + 'save/'
VECTORS_SAVE_DIR = ROOT_DIR + 'save/mismatch_vectors/'
KERNELS_SAVE_DIR = ROOT_DIR + 'save/kernel_matrices/'
TRAINING_SAVE_DIR = ROOT_DIR + 'save/trained_classifiers/cross_validation/'
LOG_PATH = ROOT_DIR + 'logs/'


# change the alphabet which will compose the k-mers to match your needs
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', ' ']
ALPHABET = [a.lower() for a in ALPHABET]

POSSIBLE_LABELS = set()
TRAINING_LIST = []
LABELS = []

# change this to match the format of your dataset making sure
# to put strings in the TRAINING_LIST list,
# corresponding labels in the LABELS list in the same order
# and in POSSIBLE_LABELS a set of every labels
with open(DATA_DIR + "examples.pk", 'rb') as examples_file:
    examples = pickle.load(examples_file)
    for _, __ in examples:
        TRAINING_LIST.append(_.lower())  # examples are not distinguished by case sensitivity
        LABELS.append(__)
        POSSIBLE_LABELS.add(__)

# do not change this
POSSIBLE_LABELS = list(POSSIBLE_LABELS)
TRAINING_LIST = np.array(TRAINING_LIST)
LABELS = np.array(LABELS)
