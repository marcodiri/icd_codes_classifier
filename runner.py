import argparse
from multiprocessing import Pool
from timeit import default_timer
from sklearn.model_selection import StratifiedKFold

from utils import *
from mismatch_kernel import MismatchKernel
from sklearn import svm
import numpy as np


class Trainer:
    def __init__(self, args):
        self.args = args
        self.MISMATCH_KERNEL = MismatchKernel(ALPHABET, self.args.k, self.args.m)
        self.mv_dir = VECTORS_SAVE_DIR
        mv_file = self.mv_dir + "mismatch_vectors_{}_{}.pk".format(self.args.k, self.args.m)
        if os.path.exists(mv_file):
            with open(mv_file, 'rb') as mismatch_vectors_file:
                self.MISMATCH_KERNEL.MISMATCH_VECTORS = pickle.load(mismatch_vectors_file)

    def _chunk_mismatch_vectors(self, pid, chk):
        size = len(chk)
        vectors_dict = {}

        # Initial call to print 0% progress
        print_progress_bar(pid=pid, iteration=0, total=size)

        for n, ex in enumerate(chk, start=1):
            # add trailing spaces to all the examples shorter than K
            ex = self.MISMATCH_KERNEL.mismatch_tree.normalize_input(ex)
            if len(ex) < self.args.k:
                ex = ex.ljust(self.args.k)
            if ex not in vectors_dict:
                ex_norm, mv = self.MISMATCH_KERNEL.vectorize(ex)
                vectors_dict[ex_norm] = mv

            # Update Progress Bar
            print_progress_bar(pid=pid, iteration=n, total=size)

        return vectors_dict

    def save_mismatch_vectors(self):
        if os.path.exists(self.mv_dir+'mismatch_vectors_{}_{}.pk'
                .format(self.args.k, self.args.m)):
            print("Mismatch_vectors file found, proceeding to next stage")
            LOGGER.info("Mismatch_vectors file found, proceeding to next stage")
            return

        processes = self.args.process_count

        print("Starting ({}-{})-mismatch-vectors calculation with {} processes"
              .format(self.args.k, self.args.m, processes))
        LOGGER.info("Starting ({}-{})-mismatch-vectors calculation with {} processes"
                    .format(self.args.k, self.args.m, processes))
        start = default_timer()

        vectors_dict = {}

        if processes > 1:
            with Pool(processes=processes) as pool:
                chunks = [pool.apply_async(func=self._chunk_mismatch_vectors,
                                           args=(pid, chk)
                                           )
                          for pid, chk in enumerate(chunk(TRAINING_LIST, processes))
                          ]

                # merge chunks in one dict
                for k in chunks:
                    vectors_dict.update(k.get())

        else:
            vectors_dict = self._chunk_mismatch_vectors(0, TRAINING_LIST)

        # add the empty label to initialize the training with a zeros vector
        # which translates to an empty dictionary in DOK format
        vectors_dict[""] = {}
        self.MISMATCH_KERNEL.MISMATCH_VECTORS = vectors_dict

        touch_dir(self.mv_dir)
        with open(self.mv_dir+'mismatch_vectors_{}_{}.pk'
                .format(self.args.k, self.args.m), 'wb')\
                as mismatch_vectors_file:
            pickle.dump(vectors_dict, mismatch_vectors_file)

        end = default_timer()
        print("Finished mismatch vectors calculation in {} seconds".format(end-start))
        LOGGER.info("Finished mismatch vectors calculation in {} seconds".format(end-start))

    def _compute_row(self, pid, chk, keys, kernel):
        rows = {}
        size = len(chk)

        # Initial call to print 0% progress
        print_progress_bar(pid=pid, iteration=0, total=size)

        for n, current_key in enumerate(chk, start=1):
            rows[current_key] = {}
            j = keys.index(current_key)
            while j < len(keys):
                other_key = keys[j]
                rows[current_key][other_key] = kernel.get_kernel(other_key, current_key)
                j += 1

            # Update Progress Bar
            print_progress_bar(pid=pid, iteration=n, total=size)

        return rows


global trainer
def train(args):
    if args.k <= args.m:
        raise RuntimeError("Length k of subsequences must be greater than the number of mismatches m")

    matrix, labels = compute_matrix(f'fold{args.fold_number}_training', args)
    print("training...")
    start = default_timer()
    clf = svm.LinearSVC(dual=False, C=args.c)
    clf.fit(matrix, labels)
    end = default_timer()
    msg = f"Finished training in {end-start} seconds"
    print(msg)
    LOGGER.info(msg)
    print("Saving classifier...")
    with open(args.savedir+f'fold{args.fold_number}_classifier.pk', 'wb') as cf:
        pickle.dump(clf, cf)


def compute_matrix(name, args):
    global trainer
    if not os.path.exists(VECTORS_SAVE_DIR+f'{name}_matrix.pk'):
        print(f"Computing {name}_matrix")
        from scipy import sparse
        size = len(args.examples)
        matrix = sparse.lil_matrix((size, len(ALPHABET)**args.k), dtype=int)
        labels = list()
        print_progress_bar(pid=0, iteration=0, total=size)
        for n, (_, __) in enumerate(args.examples):
            # add trailing spaces to all the examples shorter than K
            ex = trainer.MISMATCH_KERNEL.mismatch_tree.normalize_input(_.lower())
            if len(ex) < args.k:
                ex = ex.ljust(args.k)
            vector = trainer.MISMATCH_KERNEL.vectorize(ex)[1]
            values = list(vector.values())
            positions = list(vector.keys())
            matrix[n, positions] = values
            labels.append(__)
            print_progress_bar(pid=0, iteration=n+1, total=size)
        matrix = matrix.tocsr()
        touch_dir(VECTORS_SAVE_DIR)
        with open(VECTORS_SAVE_DIR+f'{name}_matrix.pk', 'wb') as mf:
            pickle.dump(matrix, mf)
    else:
        print(f"{name}_matrix found")
        with open(VECTORS_SAVE_DIR+f'{name}_matrix.pk', 'rb') as mf:
            matrix = pickle.load(mf)
        labels = [y for _, y in args.examples]
    return matrix, labels


def cross_validate(args):
    msg = f"LinearSVC - C={args.c}"
    print(msg)
    LOGGER.info(msg)
    from configs import POSSIBLE_LABELS, TRAINING_LIST, LABELS
    global trainer
    # k-fold cross-validation
    from sklearn.model_selection import KFold
    import numpy as np
    # data sample
    data = np.array(examples)
    # prepare cross validation
    args.splits = 4
    args.shuffle = True
    args.seed = 1
    args.savedir = TRAINING_SAVE_DIR + f"{args.splits}-fold_C{args.c}_results/"
    touch_dir(args.savedir)
    kfold = StratifiedKFold(n_splits=args.splits, shuffle=args.shuffle, random_state=args.seed)
    LOGGER.info(f"Starting {args.splits}-fold cross validation with shuffle {args.shuffle} and seed {args.seed}")
    print(f"Starting {args.splits}-fold cross validation with shuffle {args.shuffle} and seed {args.seed}")

    trainer = Trainer(args)
    trainer.save_mismatch_vectors()

    # enumerate splits
    for n, (train_indexes, test_indexes) in enumerate(kfold.split(TRAINING_LIST, LABELS)):
        args.fold_number = n
        args.examples = data[train_indexes]

        if not os.path.exists(args.savedir+f'fold{args.fold_number}_classifier.pk'):
            LOGGER.info(f"Beginning training for fold {n}")
            print(f"Beginning training for fold {n}")
            train(args)
        else:
            LOGGER.info(f"Model found for fold {n}, skipping training...")
            print(f"Model found for fold {n}, skipping training...")

        with open(args.savedir+f'fold{args.fold_number}_classifier.pk', 'rb') as sf:
            clf = pickle.load(sf)

        args.examples = data[test_indexes]
        matrix, real_labels = compute_matrix(f'fold{args.fold_number}_predict', args)

        LOGGER.info(f"Beginning predictions for fold {n}")
        print(f"Beginning predictions for fold {n}")
        start = default_timer()
        predictions = list(zip(clf.predict(matrix), real_labels))
        end = default_timer()
        msg = f"Finished predicting in {end-start} seconds"
        print(msg)
        LOGGER.info(msg)

        correct, mistaken = 0, 0
        y_true, y_pred = [], []
        for predicted, actual in predictions:
            y_true.append(actual)
            y_pred.append(predicted)
            if predicted == actual:
                correct += 1
            else:
                mistaken += 1

        from sklearn import metrics
        result = {
            "confusion_matrix": metrics.confusion_matrix(y_true, y_pred, labels=POSSIBLE_LABELS),
            "labels": POSSIBLE_LABELS,
            "y_pred": y_pred,
            "y_true": y_true
        }
        accuracy = metrics.accuracy_score(y_true, y_pred)*100
        with open(args.savedir+f"linear_fold{n}_accuracy{round(accuracy)}.pk", "wb") as res_f:
            pickle.dump(result, res_f)
        info = f"Linear prediction - Fold {n}: correct: {correct}, mistaken: {mistaken}, " \
               f"accuracy: {accuracy}\n"
        print(info)
        LOGGER.info(info)


def main():
    """
    Define and parse command line arguments.
    """
    # Create the top-level parser.
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--process_count',
                        help='Number of worker processes to use.',
                        type=int,
                        default=os.cpu_count())

    subparsers = parser.add_subparsers(help='sub-command help')

    # Create the parser for the train command.
    parser_train = subparsers.add_parser('train',
                                         help='Create and train a MulticlassClassifier')
    parser_train.add_argument('-c',
                              help='regularization parameter.',
                              type=float,
                              metavar='{0.1, 0.2, ..., 10}',
                              default=1)
    parser_train.add_argument('-k',
                              help='length of k-mers for the (k-m)-mismatch vector.',
                              type=int,
                              choices=np.array(range(2, 11)),
                              metavar='{2, ..., 10}',
                              default=2)
    parser_train.add_argument('-m',
                              help='maximum number of mismatches for the (k-m)-mismatch vector.',
                              type=int,
                              choices=np.array(range(1, 11)),
                              metavar='{1, 2, ..., 10}',
                              default=1)
    parser_train.set_defaults(func=cross_validate)

    # Parse arguments and call appropriate function (train or test).
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()

