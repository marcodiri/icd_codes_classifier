import os
import pickle
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/'
DATA_DIR = ROOT_DIR + 'data/'
MODEL_SAVE_DIR = ROOT_DIR + 'save/'
TRAINING_SAVE_DIR = ROOT_DIR + 'save/' + DATA_DIR.split('/')[-1]
LOG_PATH = ROOT_DIR + 'logs/'


# the alphabet which will compose the k-mers
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', ' ']
ALPHABET = [a.lower() for a in ALPHABET]

# change this to match the domain size of your dataset labels
POSSIBLE_LABELS = set()
TRAINING_LIST = []
LABELS = []
with open(DATA_DIR + "examples.pk", 'rb') as mismatch_vectors_file:
    examples = pickle.load(mismatch_vectors_file)
    for t, l in examples:
        TRAINING_LIST.append(t)
        LABELS.append(l)
        POSSIBLE_LABELS.add(l)

POSSIBLE_LABELS = list(POSSIBLE_LABELS)
TRAINING_LIST = np.array(TRAINING_LIST)
LABELS = np.array(LABELS)
