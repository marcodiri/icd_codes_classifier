import logging
import random
import sys
from math import floor

from configs import *


# Print iterations progress
def print_progress_bar(pid, iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    output = ""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    pid = str(pid)
    if len(pid) == 1:
        pid += " "
    output += f'#{pid}{prefix} |{bar}| {percent}% {suffix}'

    print(f"\r{output}", end="")
    # Print New Line on Complete
    if iteration == total:
        print()


def knuth_shuffle(tup, seed=None):
    """
    Shuffles locally (i.e. the lists will be modified) the passed lists
    with a Knuth shuffle. The same permutation will occur on every list.
    Lists in the tuple must have the same length

    If you want to pass only one list create a tuple like so:

    lst = [1, 2, 3]

    knuth_shuffle((lst,))

    or

    shuffled_lst = knuth_shuffle((lst.copy(),))[0] if you don't want the list
    to be modified.

    :param tup: a tuple of lists to be shuffled
    :param seed: a seed to generate repeatable permutations
    :return: the input tuple and the seed used
    """

    # create a seed
    if seed is None:
        seed = random.randrange(sys.maxsize)
    random.seed(seed)
    start_index = len(tup[0]) - 1
    while start_index > 0:
        random_index = random.randint(0, start_index)
        for l in tup:
            temp = l[start_index]
            l[start_index] = l[random_index]
            l[random_index] = temp
        start_index -= 1
    return tup, seed


# split mismatch vectors list
def chunk(lst, chunks):
    """
    Generate chunks equal parts of lst, if len(lst) is not divisible
    by chunks, the last chunk will contain the remaining elements.
    :param lst: the list to be chunked
    :param chunks: number of chunks to split the list into
    """
    elements_per_chunk = floor(len(lst) / chunks)
    for i in range(0, len(lst), elements_per_chunk):
        if i < elements_per_chunk * (chunks - 1):
            if isinstance(lst, dict):
                subset = list(lst.keys())[i:i + elements_per_chunk]
                yield {key: lst[key] for key in subset}
            else:
                yield lst[i:i + elements_per_chunk]
        else:
            if isinstance(lst, dict):
                subset = list(lst.keys())[i:]
                yield {key: lst[key] for key in subset}
            else:
                yield lst[i:]
            return


def touch_dir(base_dir: str) -> None:
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)


def _get_logger(name: str):
    touch_dir(LOG_PATH)
    logging.basicConfig(filename=LOG_PATH + 'events.log',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S'
                        )
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    return logger


LOGGER = _get_logger(__name__)
