from multiprocessing import Pool, Manager
from timeit import default_timer
from utils import *
from mismatch_kernel import MismatchKernel


class MulticlassClassifier:
    """
    Manages the BinaryClassifiers
    """
    def __init__(self, possible_labels, BinaryClassifier, args, kernel):
        self.args = args

        self.kernel = kernel
        self.binary_classifier = BinaryClassifier
        self.binary_classifiers = {y:
                                   BinaryClassifier(self.kernel)
                                   for y in possible_labels}

    def normalize_labels(self, labels, eval_label):
        """
        Normalize the labels list to contain only 1 or -1 based on the evaluating label
        :param labels: list of labels corresponding to the training set
        :param eval_label: label to be trained for by the BinaryClassifier
        :return: the normalized list of labels
        """
        return np.where(np.isin(labels, eval_label),
                        np.ones(labels.shape, np.int8),
                        -np.ones(labels.shape, np.int8))

    def _chunk_binary_classifiers_train(self, pid, progress_list, chunk, training_list, training_labels):
        size = len(chunk)

        # Initial call to print 0% progress
        progress_list.insert(pid, [0, size])
        print_progress_bar(progress_list)

        for n, label in enumerate(chunk, start=1):
            binary_classifier = chunk[label]
            normalized_labels = self.normalize_labels(training_labels, label)
            binary_classifier.train(training_list, normalized_labels)

            # Update Progress Bar
            progress_list[pid] = [n, size]
            print_progress_bar(progress_list)

        return chunk

    def train(self, training_list, labels):
        # if kernel is (K,M)-Mismatch add trailing spaces to all the
        # examples shorter than K
        if isinstance(self.kernel, MismatchKernel):
            for i, v in enumerate(training_list):
                v = self.kernel.mismatch_tree.normalize_input(v)
                if len(v) < self.args.k:
                    training_list[i] = v.ljust(self.args.k)

        print("Starting training")
        LOGGER.info("Starting training")
        start = default_timer()

        if self.args.process_count is not None and self.args.process_count > 1:
            with Pool(processes=self.args.process_count) as pool:
                progress_list = Manager().list()
                chunks = [pool.apply_async(func=self._chunk_binary_classifiers_train,
                                           args=(pid, progress_list, chk, training_list, labels))
                          for pid, chk in enumerate(chunk(self.binary_classifiers, self.args.process_count))]

                # retrieve the trained binary classifiers back from the process pool and
                # replace the untrained instances in self.binary_classifiers with
                # their trained counterparts
                for k in chunks:
                    self.binary_classifiers.update(k.get())

        else:
            size = len(self.binary_classifiers)
            # Initial call to print 0% progress
            print_progress_bar([[0, size]], length=50)
            # train each binary classifier with a single process
            for n, (label, binary_classifier) in enumerate(self.binary_classifiers.items(), start=1):
                normalized_labels = self.normalize_labels(labels, label)
                binary_classifier.train(training_list, normalized_labels)
                # print("Finished training for class {}/{} - {}".format(n, len(self.binary_classifiers), label))
                print_progress_bar([[n, size]], length=50)

        end = default_timer()

        # return number of prediction vectors making up each binary classifier
        bc_vector_counts = [(k, len(v.weights))
                            for k, v in self.binary_classifiers.items()]
        tot_errors = sum(e for c, e in bc_vector_counts)

        LOGGER.info("Training time: {} sec".format(end-start))
        print("Training time: {} sec".format(end-start))
        LOGGER.info("Per class error distribution:")
        LOGGER.info("{}".format(bc_vector_counts))
        LOGGER.info("Total errors: {}".format(tot_errors))
        print("Total errors: {}".format(tot_errors))

        # using more processes generates larger VotedPerceptron objects
        # in terms of memory usage, so recreate them with same
        # attributes to compress the MulticlassClassifier
        if self.args.process_count > 1:
            print("Compressing MulticlassClassifier")
            for _, __ in self.binary_classifiers.items():
                temp = self.binary_classifier(self.kernel)
                temp.mistaken_examples = __.mistaken_examples
                temp.mistaken_labels = __.mistaken_labels
                temp.weights = __.weights
                temp.w = __.w
                # temp.current_weight = __.current_weight  # not necessary
                self.binary_classifiers[_] = temp

        # save trained MulticlassClassifier
        print('Saving MulticlassClassifier')
        training_dir = TRAINING_SAVE_DIR
        touch_dir(training_dir)
        save_filepath = training_dir+'/{}_{}_{}_fold{}_{}_{}_{}_epochs{}.pk'\
            .format(self.args.splits, self.args.shuffle, self.args.seed, self.args.fold_number,
                    self.kernel.__class__.__name__, self.args.k, self.args.m, self.args.epochs)

        with open(save_filepath, 'wb') as multicc_file:
            pickle.dump(self, multicc_file)
        LOGGER.info("Created save file in {}\n".format(save_filepath))

    def predict(self, x):
        # Note: not multiprocess because the overhead of having
        # multiple processes overwhelms the time to make the calculation

        # if kernel is (K,M)-Mismatch add trailing spaces to x
        if isinstance(self.kernel, MismatchKernel):
            x = x.lower()
            x = self.kernel.mismatch_tree.normalize_input(x)
            if len(x) < self.args.k:
                x = x.ljust(self.args.k)
        bc_scores = {label: binary_classifier.predict(x)
                     for label, binary_classifier in self.binary_classifiers.items()}

        sorted_scores = sorted(bc_scores.items(), key=lambda x: x[1], reverse=True)
        # print(f"First 3 scores: {sorted_scores[:3]}")
        return sorted_scores[0][0]

