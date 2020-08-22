import sys
sys.path.append('D:\\Users\\Marco\\Google Drive\\Uni\\Tesi\\mismatch_string_kernel')  # needed to run script from console
import argparse
from math import floor
from multiprocessing import Pool
from timeit import default_timer

from utils import *
from votedperceptron import VotedPerceptron, MulticlassClassifier
from mismatch_kernel import MismatchKernel


class Trainer:
    def __init__(self, args):
        self.args = args
        self.MISMATCH_KERNEL = MismatchKernel(ALPHABET, self.args.k, self.args.m)
        self.mv_dir = TRAINING_SAVE_DIR + "mismatch_vectors/"
        mv_file = self.mv_dir + "mismatch_vectors_{}_{}.pk".format(self.args.k, self.args.m)
        if os.path.exists(mv_file):
            with open(mv_file, 'rb') as mismatch_vectors_file:
                self.MISMATCH_KERNEL.MISMATCH_VECTORS = pickle.load(mismatch_vectors_file)

        self.km_dir = TRAINING_SAVE_DIR + "kernel_matrices/"
        km_file = self.km_dir + "kernel_matrix_{}_{}.pk".format(self.args.k, self.args.m)
        if os.path.exists(km_file):
            with open(km_file, 'rb') as kernel_matrix_file:
                self.MISMATCH_KERNEL.KERNEL_MATRIX = pickle.load(kernel_matrix_file)

    def _chunk_mismatch_vectors(self, pid, chunk):
        size = len(chunk)-1
        vectors_dict = {}

        # Initial call to print 0% progress
        print_progress_bar(0, size,
                           prefix='#{} Progress:'.format(pid), suffix='Complete', length=50)

        for n, (t, l) in enumerate(chunk):
            # add trailing spaces to all the examples shorter than K
            if len(t) < self.args.k:
                t = t.ljust(self.args.k)
            if t not in vectors_dict:
                t_norm, mv = self.MISMATCH_KERNEL.mismatch_tree.vectorize(t)
                vectors_dict[t_norm] = mv

            # Update Progress Bar
            print_progress_bar(n, size,
                               prefix='#{} Progress:'.format(pid), suffix='Complete', length=50)

        return vectors_dict

    def save_mismatch_vectors(self):
        if os.path.exists(self.mv_dir+'mismatch_vectors_{}_{}.pk'
                .format(self.args.k, self.args.m)):
            print("Mismatch_vectors file found, proceeding to next stage")
            return

        print("Starting ({}-{})-mismatch-vectors calculation".format(self.args.k, self.args.m))
        start = default_timer()

        processes = self.args.process_count
        vectors_dict = {}

        if processes > 1:
            # split examples list
            def chunk(lst):
                elements_per_chunk = floor(len(lst)/processes)
                for i in range(0, len(lst), elements_per_chunk):
                    if i < elements_per_chunk*(processes-1):
                        yield lst[i:i+elements_per_chunk]
                    else:
                        yield lst[i:]
                        return

            with Pool(processes=processes) as pool:
                chunks = [pool.apply_async(func=self._chunk_mismatch_vectors,
                                           args=(pid, c)
                                           )
                          for pid, c in enumerate(chunk(examples))
                          ]

                # merge chunks in one dict
                for k in chunks:
                    vectors_dict.update(k.get())

        else:
            vectors_dict = self._chunk_mismatch_vectors(0, examples)

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

    def _compute_row(self, pid, chunk, keys, kernel):
        rows = {}
        size = len(chunk)-1
        # Initial call to print 0% progress
        print_progress_bar(0, size,
                           prefix='Progress:', suffix='Complete', length=50)
        for n, current_key in enumerate(chunk):
            rows[current_key] = {}
            j = keys.index(current_key)
            while j < len(keys):
                other_key = keys[j]
                rows[current_key][other_key] = kernel.get_kernel(other_key, current_key)
                j += 1

            # Update Progress Bar
            print_progress_bar(n, size,
                               prefix='#{} Progress:'.format(pid), suffix='Complete', length=50)
        return rows

    def kernel_matrix(self):
        processes = self.args.process_count
        if os.path.exists(self.km_dir+'kernel_matrix_{}_{}.pk'
                .format(self.args.k, self.args.m)):
            print("Kernel_matrix file found, proceeding to next stage")
            return

        print("Starting kernel matrix calculation")
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
                rows_chunks = [pool.apply_async(func=self._compute_row,
                                                args=(pid, c, keys, self.MISMATCH_KERNEL)
                                                )
                               for pid, c in enumerate(chunk(knuth_shuffle((keys.copy(),))[0], processes))
                               ]

                for k in rows_chunks:
                    matrix.update(k.get())
        else:
            size = len(keys)-1
            # Initial call to print 0% progress
            print_progress_bar(0, size,
                               prefix='Progress:', suffix='Complete', length=50)
            for i, k in enumerate(keys):
                j = i
                while j < len(mv):
                    current_key = keys[j]
                    if k not in matrix:
                        matrix[k] = {}
                    matrix[k][current_key] = self.MISMATCH_KERNEL.get_kernel(k, current_key)
                    j += 1

                # Update Progress Bar
                print_progress_bar(i, size,
                                   prefix='Progress:', suffix='Complete', length=50)

        self.MISMATCH_KERNEL.KERNEL_MATRIX = matrix

        touch_dir(self.km_dir)
        with open(self.km_dir+'kernel_matrix_{}_{}.pk'
                .format(self.args.k, self.args.m), 'wb')\
                as km_file:
            pickle.dump(matrix, km_file)

        end = default_timer()
        print("Finished mismatch vectors calculation in {} seconds".format(end-start))

    def train(self):
        def expand_dataset(lst):
            """
            :param lst: the list to be shuffled
            :return: a copy of lst with shuffled elements
            """
            expanded = [l.copy() for l in lst]
            for e in range(1, self.args.epochs):
                shuffled = knuth_shuffle([l.copy() for l in lst])
                # attach shuffled lists to original lists
                for i in range(len(lst)):
                    expanded[i] += shuffled[i]
            return expanded

        print("Starting training for {} epochs".format(self.args.epochs))
        start = default_timer()

        # WARNING: for some reason using more processes produces a much bigger file
        dataset = expand_dataset([list(TRAINING_LIST), list(LABELS)])

        # create instance of MulticlassClassifier
        multicc = MulticlassClassifier(POSSIBLE_LABELS, VotedPerceptron, self.args, self.MISMATCH_KERNEL)
        multicc.train(np.array(dataset[0]), np.array(dataset[1]))

        end = default_timer()
        print("Finished training {} seconds".format(end-start))


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
    trainer.train()
    pass


def predict(args):
    with open(args.filepath, 'rb') as file:
        mcc = pickle.load(file)
        print("K={} {}epochs guessed {}\n".format(mcc.args.k, mcc.args.epochs, mcc.predict(args.input)))


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
    parser_train.set_defaults(func=train)

    # Create the parser for the test command.
    parser_test = subparsers.add_parser('predict',
                                        help='Predict an input with a trained MulticlassClassifier')
    parser_test.add_argument('-i', '--input',
                             help='The input to predict.',
                             type=str
                             )
    parser_test.add_argument('-f', '--filepath',
                             help='Training file to use to predict the input.',
                             type=str
                             )
    parser_test.set_defaults(func=predict)

    # Parse arguments and call appropriate function (train or test).
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    # with open("save/kernel_matrices/kernel_matrix_2_1.pk", "rb") as fsd:
    #     asd = pickle.load(fsd)
    #     pass
    main()
    # from argparse import Namespace
    # args = Namespace()
    # args.input = "diarrea"
    # args.filepath = TRAINING_SAVE_DIR+"trained_classifiers/MismatchKernel_2_1_epochs1_errors8602.pk"
    # predict(args)
    # args.filepath = TRAINING_SAVE_DIR+"trained_classifiers/MismatchKernel_3_1_epochs1_errors8369.pk"
    # predict(args)
    # args.filepath = TRAINING_SAVE_DIR+"trained_classifiers/MismatchKernel_3_1_epochs3_errors16099.pk"
    # predict(args)
    # args.filepath = TRAINING_SAVE_DIR+"trained_classifiers/MismatchKernel_4_1_epochs1_errors14621.pk"
    # predict(args)
    # start = default_timer()
    # # save_mismatch_vectors()
    # # kernel_matrix()
    # # kernel_matrix(1)  # reduce file size
    # # train()
    # predict("scomp. cadrico cong")  # solo 4
    # # predict("diarrea")  # 3 2epoche a 4
    # # predict("grave bradiaritmia")  # solo 3 3epoche e 4
    # # predict("scopeso cadrico")  # solo 2
    # predict("fratt femore")  # tutti tranne 4
    # # predict("cad")  # solo 3 3epoche a 4
    # # predict("ins ren")  # solo 3 fino a 3epoche
    # end = default_timer()
    # print("Time {}".format(end-start))

