from multiprocessing import Pool
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

    def _chunk_binary_classifiers_train(self, pid, chunk, training_list, training_labels):
        size = len(chunk)

        # Initial call to print 0% progress
        print_progress_bar(pid=pid, iteration=0, total=size)

        for n, label in enumerate(chunk, start=1):
            binary_classifier = chunk[label]
            normalized_labels = self.normalize_labels(training_labels, label)
            binary_classifier.train(training_list, normalized_labels)

            # Update Progress Bar
            print_progress_bar(pid=pid, iteration=n, total=size)

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
                chunks = [pool.apply_async(func=self._chunk_binary_classifiers_train,
                                           args=(pid, chk, training_list, labels))
                          for pid, chk in enumerate(chunk(self.binary_classifiers, self.args.process_count))]

                # retrieve the trained binary classifiers back from the process pool and
                # replace the untrained instances in self.binary_classifiers with
                # their trained counterparts
                for k in chunks:
                    self.binary_classifiers.update(k.get())

        else:
            size = len(self.binary_classifiers)
            # Initial call to print 0% progress
            print_progress_bar(pid=0, iteration=0, total=size)
            # train each binary classifier with a single process
            for n, (label, binary_classifier) in enumerate(self.binary_classifiers.items(), start=1):
                normalized_labels = self.normalize_labels(labels, label)
                binary_classifier.train(training_list, normalized_labels)
                # print("Finished training for class {}/{} - {}".format(n, len(self.binary_classifiers), label))
                print_progress_bar(pid=0, iteration=n, total=size)

        end = default_timer()

        LOGGER.info("Training time: {} sec".format(end-start))
        print("Training time: {} sec".format(end-start))

    def predict(self, x, mode):
        # Note: not multiprocess because the overhead of having
        # multiple processes overwhelms the time to make the calculation

        # if kernel is (K,M)-Mismatch add trailing spaces to x
        if isinstance(self.kernel, MismatchKernel):
            x = x.lower()
            x = self.kernel.mismatch_tree.normalize_input(x)
            if len(x) < self.args.k:
                x = x.ljust(self.args.k)
        if mode == "single":
            bc_scores = {label: binary_classifier.predict_single(x)
                         for label, binary_classifier in self.binary_classifiers.items()}
        else:
            bc_scores = {label: binary_classifier.predict_voted(x)
                         for label, binary_classifier in self.binary_classifiers.items()}

        sorted_scores = sorted(bc_scores.items(), key=lambda x: x[1], reverse=True)
        # print(f"First 3 scores: {sorted_scores[:3]}")
        return sorted_scores[0][0]

