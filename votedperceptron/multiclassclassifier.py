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
        size = len(chunk)-1
        # Initial call to print 0% progress
        print_progress_bar(0, size,
                           prefix='#{} Progress:'.format(pid), suffix='Complete', length=50)
        for n, label in enumerate(chunk):
            binary_classifier = chunk[label]
            normalized_labels = self.normalize_labels(training_labels, label)
            binary_classifier.train(training_list, normalized_labels)
            print_progress_bar(n, size,
                               prefix='#{} Progress:'.format(pid), suffix='Complete', length=50)

        return chunk

    def train(self, training_list, labels):
        # if kernel is (K,M)-Mismatch add trailing spaces to all the
        # examples shorter than K
        if isinstance(self.kernel, MismatchKernel):
            for i, v in enumerate(training_list):
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
            # Initial call to print 0% progress
            print_progress_bar(0, len(self.binary_classifiers), prefix='Progress:', suffix='Complete', length=50)
            # train each binary classifier with a single process
            for n, (label, binary_classifier) in enumerate(self.binary_classifiers.items()):
                normalized_labels = self.normalize_labels(labels, label)
                binary_classifier.train(training_list, normalized_labels)
                # print("Finished training for class {}/{} - {}".format(n, len(self.binary_classifiers), label))
                print_progress_bar(n, len(self.binary_classifiers), prefix='Progress:', suffix='Complete', length=50)

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
                # temp.current_weight = __.current_weight  # not necessary
                self.binary_classifiers[_] = temp

        # save trained MulticlassClassifier
        print('Saving MulticlassClassifier')
        training_dir = TRAINING_SAVE_DIR
        touch_dir(training_dir)
        save_filepath = training_dir+'/{}_{}_{}_epochs{}_errors{}.pk'\
            .format(self.kernel.__class__.__name__, self.args.k, self.args.m, self.args.epochs, tot_errors)

        with open(save_filepath, 'wb') as multicc_file:
            pickle.dump(self, multicc_file)
        LOGGER.info("Created save file in {}\n".format(save_filepath))

    def predict(self, input_vector):
        # Note: not multiprocess because the overhead of having
        # multiple processes overwhelms the time to make the calculation

        # if kernel is (K,M)-Mismatch add trailing spaces to input
        if isinstance(self.kernel, MismatchKernel):
            if len(input_vector) < self.args.k:
                input_vector = input_vector.ljust(self.args.k)
        bc_scores = {label: binary_classifier.predict(input_vector)
                     for label, binary_classifier in self.binary_classifiers.items()}

        sorted_scores = sorted(bc_scores.items(), key=lambda x: x[1], reverse=True)
        print("First 3 scores:")
        print(sorted_scores[:3])
        return sorted_scores[0][0]

