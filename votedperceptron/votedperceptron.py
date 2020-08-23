from itertools import accumulate
from math import copysign


class VotedPerceptron:
    """
    Represent one of the possible labels
    """
    def __init__(self, kernel):
        self.kernel = kernel

        # initialize structures that will store the prediction vectors
        # to later compute predictions in O(k) kernel calculations
        # with k = len(mistaken_examples) (number of errors during training)
        self.mistaken_examples = []
        self.mistaken_labels = []

        # prediction vector votes generated during training
        self.weights = []

    def train(self, training_list, labels):
        # initialize structures
        if not self.mistaken_examples:
            self.current_weight = 0
            init_example = ""  # the empty string corresponds to the zeros vector
            self.mistaken_examples.append(init_example)
            self.mistaken_labels.append(1)
        else:
            # if there are more examples the weight saved at
            # the end of the last chunk is incorrect
            # (it was only needed to save the intermediate epoch)
            self.weights.pop()
        ind = -1
        for x, y_real in zip(training_list, labels):
            ind += 1
            # computing the prediction is the slow part.
            # It does O(n_examples * k^2) kernel calculations
            # with k number of mistakes made during the training
            prediction = sum(ml * self.kernel.get_kernel(me, x)
                             for me, ml
                             in zip(self.mistaken_examples, self.mistaken_labels)
                             )

            y_predicted = copysign(1, prediction)

            if y_predicted == y_real:  # correct prediction
                self.current_weight += 1
            else:  # wrong prediction
                # save mistaken example and label and weight
                self.mistaken_examples.append(x.copy())
                self.mistaken_labels.append(y_real)
                self.weights.append(self.current_weight)
                self.current_weight = 1  # reset weight

        # training complete
        # save the last weight
        self.weights.append(self.current_weight)

    def predict(self, x):
        pv_pre_activations = accumulate(yi * self.kernel.get_kernel(xi, x)
                                        for yi, xi
                                        in zip(self.mistaken_labels,
                                               self.mistaken_examples))

        pre_activation = sum(
            w
            * copysign(1, pvpa)
            for w, pvpa
            in zip(self.weights, pv_pre_activations)
        )

        return pre_activation
