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
