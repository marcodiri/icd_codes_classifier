import argparse
from multiprocessing import Pool, Manager
from timeit import default_timer
from sklearn.model_selection import StratifiedKFold

from utils import *
from votedperceptron import VotedPerceptron, MulticlassClassifier
from mismatch_kernel import MismatchKernel


class Trainer:
    def __init__(self, args):
        self.args = args
        self.MISMATCH_KERNEL = MismatchKernel(ALPHABET, self.args.k, self.args.m)
        self.mv_dir = VECTORS_SAVE_DIR
        mv_file = self.mv_dir + "mismatch_vectors_{}_{}.pk".format(self.args.k, self.args.m)
        if os.path.exists(mv_file):
            with open(mv_file, 'rb') as mismatch_vectors_file:
                self.MISMATCH_KERNEL.MISMATCH_VECTORS = pickle.load(mismatch_vectors_file)

        self.km_dir = KERNELS_SAVE_DIR
        km_file = self.km_dir + "kernel_matrix_{}_{}.pk".format(self.args.k, self.args.m)
        if os.path.exists(km_file):
            with open(km_file, 'rb') as kernel_matrix_file:
                self.MISMATCH_KERNEL.KERNEL_MATRIX = pickle.load(kernel_matrix_file)

    def _chunk_mismatch_vectors(self, pid, progress_list, chk):
        size = len(chk)
        vectors_dict = {}

        # Initial call to print 0% progress
        progress_list.insert(pid, [0, size])
        print_progress_bar(progress_list)

        for n, ex in enumerate(chk, start=1):
            # add trailing spaces to all the examples shorter than K
            ex = self.MISMATCH_KERNEL.mismatch_tree.normalize_input(ex)
            if len(ex) < self.args.k:
                ex = ex.ljust(self.args.k)
            if ex not in vectors_dict:
                ex_norm, mv = self.MISMATCH_KERNEL.vectorize(ex)
                vectors_dict[ex_norm] = mv

            # Update Progress Bar
            progress_list[pid] = [n, size]
            print_progress_bar(progress_list)

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
            progress_list = Manager().list()
            with Pool(processes=processes) as pool:
                chunks = [pool.apply_async(func=self._chunk_mismatch_vectors,
                                           args=(pid, progress_list, chk)
                                           )
                          for pid, chk in enumerate(chunk(TRAINING_LIST, processes))
                          ]

                # merge chunks in one dict
                for k in chunks:
                    vectors_dict.update(k.get())

        else:
            vectors_dict = self._chunk_mismatch_vectors(0, [], TRAINING_LIST)

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

    def _compute_row(self, pid, progress_list, chk, keys, kernel):
        rows = {}
        size = len(chk)

        # Initial call to print 0% progress
        progress_list.insert(pid, [0, size])
        print_progress_bar(progress_list)

        for n, current_key in enumerate(chk, start=1):
            rows[current_key] = {}
            j = keys.index(current_key)
            while j < len(keys):
                other_key = keys[j]
                rows[current_key][other_key] = kernel.get_kernel(other_key, current_key)
                j += 1

            # Update Progress Bar
            progress_list[pid] = [n, size]
            print_progress_bar(progress_list)

        return rows

    def kernel_matrix(self):
        processes = self.args.process_count
        if os.path.exists(self.km_dir+'kernel_matrix_{}_{}.pk'
                .format(self.args.k, self.args.m)):
            print("Kernel_matrix file found, proceeding to next stage")
            LOGGER.info("Kernel_matrix file found, proceeding to next stage")
            return

        print("Starting ({}-{})-kernel-matrix calculation with {} processes"
              .format(self.args.k, self.args.m, processes))
        LOGGER.info("Starting ({}-{})-kernel-matrix calculation with {} processes"
                    .format(self.args.k, self.args.m, processes))
        start = default_timer()

        matrix = {}
        mv = self.MISMATCH_KERNEL.MISMATCH_VECTORS
        keys = list(mv.keys())
        if processes > 1:
            # each process computes a chunk of rows of the matrix.
            # Since the keys towards the end of the list require
            # much less calculation, the keys list is shuffled
            # before chunking so every process gets about the
            # same amount of work
            with Pool(processes=processes) as pool:
                progress_list = Manager().list()
                rows_chunks = [pool.apply_async(func=self._compute_row,
                                                args=(pid, progress_list, chk, keys, self.MISMATCH_KERNEL)
                                                )
                               for pid, chk in enumerate(chunk(knuth_shuffle((keys.copy(),))[0][0], processes))
                               ]

                for k in rows_chunks:
                    matrix.update(k.get())
        else:
            size = len(keys)
            # Initial call to print 0% progress
            print_progress_bar([[0, size]], length=50)
            for i, k in enumerate(keys, start=1):
                j = i
                while j < len(mv):
                    current_key = keys[j]
                    if k not in matrix:
                        matrix[k] = {}
                    matrix[k][current_key] = self.MISMATCH_KERNEL.get_kernel(k, current_key)
                    j += 1

                # Update Progress Bar
                print_progress_bar([[i, size]], length=50)

        self.MISMATCH_KERNEL.KERNEL_MATRIX = matrix

        touch_dir(self.km_dir)
        with open(self.km_dir+'kernel_matrix_{}_{}.pk'
                .format(self.args.k, self.args.m), 'wb')\
                as km_file:
            pickle.dump(matrix, km_file)

        end = default_timer()
        print("Finished mismatch vectors calculation in {} seconds".format(end-start))
        LOGGER.info("Finished mismatch vectors calculation in {} seconds".format(end-start))

    def train(self, seeds=None):
        """
        :param seeds: a list of seeds to repeat dataset random permutations
        """
        if seeds is None:
            seeds = [None for _ in range(self.args.epochs-1)]
        if not isinstance(seeds, list):
            seeds = [seeds]
        if len(seeds) < self.args.epochs-1:
            raise RuntimeError("Given less seeds than epochs-1")

        def expand_dataset(lst):
            """
            :param lst: the list to be shuffled
            :return: a copy of lst with shuffled elements
            """
            print("Expanding dataset for {} epochs".format(self.args.epochs))
            LOGGER.info("Expanding dataset for {} epochs".format(self.args.epochs))
            expanded = [_.copy() for _ in lst]
            for e in range(1, self.args.epochs):
                shuffled, seed = knuth_shuffle([_.copy() for _ in lst], seeds[e-1])
                print("Shuffle {} with seed {}".format(e, seed))
                LOGGER.info("Shuffle {} with seed {}".format(e, seed))
                # attach shuffled lists to original lists
                for i in range(len(lst)):
                    expanded[i] += shuffled[i]
            return expanded

        dataset = expand_dataset([list(TRAINING_LIST), list(LABELS)])

        # create instance of MulticlassClassifier
        multicc = MulticlassClassifier(POSSIBLE_LABELS, VotedPerceptron, self.args, self.MISMATCH_KERNEL)
        multicc.train(np.array(dataset[0]), np.array(dataset[1]))
        return multicc


def train(args):
    # larger K makes the kernel stricter.
    # For example with K=2, the normalized kernel between
    # "fibrillazione atriale" and "fibrillazione atriale cronica"
    # is 0.962 which means it thinks they are very similar,
    # while with K=3 it gives 0.878.
    # So if you have available a lot of examples a smaller K can
    # work fine and makes the computation faster,
    # on the other hand if you have a limited number of examples
    # you may require a larger K to achieve an acceptable prediction rate
    # on similar strings like those above and typos.
    # For example in the ICD codes classifier the dataset was small
    # and with K=2, "fibrillazione atriale" and "fibrillazione atriale cronica"
    # were both classified as "I489", while with K=3 they got correctly
    # classified as "I489" and "I482" respectively; meaning that with K=2
    # the number of examples was not enough to compensate the large similarity
    # (0.962) between the two.
    # On the other hand, bad typos like "scopeso cadrico", which is a terribly typoed "scompenso cardiaco",
    # with K=2 gets correctly classified as "I509",
    # while with K=3 and K=4 it's wrongly classified, because the
    # typos are so dense that a 3-mer or 4-mer are too wide to catch any useful substring
    # (typing instead "scompeso cadrico", notice the extra m, already classifies correctly
    # with both K=3 and K=4).
    # Also some strings that appears in the dataset only once gets classified correctly
    # sometimes by K=3, sometimes by K=4. So there's no accurate way to classify those.
    # Overall, K=3 3epochs seems to work on most entries and it's faster.

    # Note: the length of the string to vectorize must be greater than or equal to K
    # and M must be less than K.
    # M=1 is the more efficient and stricter.

    # Note: with more than 1 epoch the order of the examples is randomized
    # so subsequent trainings will result in different quality classifiers

    if args.k <= args.m:
        raise RuntimeError("Length k of subsequences must be greater than the number of mismatches m")

    trainer = Trainer(args)
    trainer.save_mismatch_vectors()
    trainer.kernel_matrix()
    trained_classifier = trainer.train(args.seeds)

    # return number of prediction vectors making up each binary classifier
    bc_vector_counts = [(k, len(v.weights))
                        for k, v in trained_classifier.binary_classifiers.items()]
    tot_errors = sum(e for c, e in bc_vector_counts)

    LOGGER.info("Per class error distribution:")
    LOGGER.info("{}".format(bc_vector_counts))
    LOGGER.info("Total errors: {}".format(tot_errors))
    print("Total errors: {}".format(tot_errors))

    # using more processes generates larger VotedPerceptron objects
    # in terms of memory usage, so recreate them with same
    # attributes to compress the MulticlassClassifier
    if args.process_count > 1:
        print("Compressing MulticlassClassifier")
        for _, __ in trained_classifier.binary_classifiers.items():
            temp = trained_classifier.binary_classifier(trained_classifier.kernel)
            temp.mistaken_examples = __.mistaken_examples
            temp.mistaken_labels = __.mistaken_labels
            temp.weights = __.weights
            temp.w = __.w
            # temp.current_weight = __.current_weight  # not necessary
            trained_classifier.binary_classifiers[_] = temp

    # save trained MulticlassClassifier
    print('Saving MulticlassClassifier')
    training_dir = TRAINING_SAVE_DIR
    touch_dir(training_dir)
    save_filepath = training_dir + '/{}_{}_{}_fold{}_{}_{}_{}_epochs{}.pk' \
        .format(args.splits, args.shuffle, args.seed, args.fold_number,
                trained_classifier.kernel.__class__.__name__, args.k, args.m, args.epochs)

    with open(save_filepath, 'wb') as multicc_file:
        pickle.dump(trained_classifier, multicc_file)
    LOGGER.info("Created save file in {}\n".format(save_filepath))


def _predict_batch(pid, progress_list, batch, mode):
    size = len(batch)
    predictions = []

    # Initial call to print 0% progress
    progress_list.insert(pid, [0, size])
    print_progress_bar(progress_list)

    for __, _ in enumerate(batch, start=1):
        predicted = mcclassifier.predict(_[0], mode)
        true = _[1]
        predictions.append({
            "string": _[0],
            "predicted": predicted,
            "true": true
        })

        # Update Progress Bar
        progress_list[pid] = [__, size]
        print_progress_bar(progress_list)

    return predictions


def init(mcc):
    global mcclassifier
    mcclassifier = mcc


def cross_validate(args):
    from configs import POSSIBLE_LABELS, TRAINING_LIST, LABELS
    global POSSIBLE_LABELS, TRAINING_LIST, LABELS
    # k-fold cross-validation
    from sklearn.model_selection import KFold
    # data sample
    data = np.array(examples)
    # prepare cross validation
    args.splits = 4
    args.shuffle = True
    args.seed = 1
    args.prediction_mode = "voted"
    kfold = StratifiedKFold(n_splits=args.splits, shuffle=args.shuffle, random_state=args.seed)
    LOGGER.info(f"Starting {args.splits}-fold cross validation with shuffle {args.shuffle} and seed {args.seed}")
    print(f"Starting {args.splits}-fold cross validation with shuffle {args.shuffle} and seed {args.seed}")
    # enumerate splits
    for n, (train_indexes, test_indexes) in enumerate(kfold.split(TRAINING_LIST, LABELS)):
        args.fold_number = n
        POSSIBLE_LABELS = set()
        TRAINING_LIST = []
        LABELS = []
        for _, __ in data[train_indexes]:
            TRAINING_LIST.append(_.lower())  # examples are not distinguished by case sensitivity
            LABELS.append(__)
            POSSIBLE_LABELS.add(__)
        POSSIBLE_LABELS = list(POSSIBLE_LABELS)
        TRAINING_LIST = np.array(TRAINING_LIST)
        LABELS = np.array(LABELS)

        savepath = TRAINING_SAVE_DIR + '/{}_{}_{}_fold{}_{}_{}_{}_epochs{}.pk' \
            .format(args.splits, args.shuffle, args.seed, args.fold_number,
                    "MismatchKernel", args.k, args.m, args.epochs)
        if not os.path.exists(savepath):
            LOGGER.info(f"Beginning training for fold {n}")
            print(f"Beginning training for fold {n}")
            train(args)
        else:
            LOGGER.info(f"Model found for fold {n}, skipping training...")
            print(f"Model found for fold {n}, skipping training...")

        with open(savepath, 'rb') as f:
            mcc = pickle.load(f)
        predictions = []
        if args.process_count > 1:
            with Pool(processes=args.process_count, initializer=init, initargs=(mcc,)) as pool:
                progress_list = Manager().list()
                results = [pool.apply_async(func=_predict_batch, args=(pid, progress_list, batch, args.prediction_mode))
                           for pid, batch in enumerate(chunk(data[test_indexes], args.process_count))]

                print("Predicting...")
                for res in results:
                    predictions += res.get()
        else:
            print("Predicting...")
            predictions = _predict_batch(0, [], data[test_indexes], args.prediction_mode)

        correct, mistaken = 0, 0
        y_true, y_pred = [], []
        for _ in predictions:
            predicted = _["predicted"]
            true = _["true"]
            y_true.append(predicted)
            y_pred.append(true)
            if predicted == true:
                correct += 1
            else:
                mistaken += 1

        from sklearn import metrics
        result = {
            "cm": metrics.confusion_matrix(y_true, y_pred, labels=POSSIBLE_LABELS),
            "labels": POSSIBLE_LABELS,
            "predictions": predictions
        }
        accuracy = metrics.accuracy_score(y_true, y_pred)*100
        savedir = TRAINING_SAVE_DIR+f"{args.splits}-fold_results/"
        touch_dir(savedir)
        if args.prediction_mode == "voted":
            with open(savedir+f"voted_fold{n}_accuracy{round(accuracy)}.pk", "wb") as res_f:
                pickle.dump(result, res_f)
            info = f"Voted prediction - Fold {n}: correct: {correct}, mistaken: {mistaken}, " \
                   f"accuracy: {accuracy}\n"
        else:
            with open(savedir+f"single_fold{n}_accuracy{round(accuracy)}.pk", "wb") as res_f:
                pickle.dump(result, res_f)
            info = f"Single prediction - Fold {n}: correct: {correct}, mistaken: {mistaken}, " \
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
    parser_train.add_argument('-e', '--epochs',
                              help='number of times the training set will be repeated.',
                              type=int,
                              choices=np.array(range(1, 11)),
                              metavar='{1, 2, ..., 10}',
                              default=1)
    parser_train.add_argument('-s', '--seeds',
                              help='number of times the training set will be repeated.',
                              type=int,
                              nargs='+',
                              metavar='[int, int, ...]',
                              default=None)
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
