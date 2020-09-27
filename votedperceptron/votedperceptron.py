from itertools import accumulate
from math import copysign


class VotedPerceptron:
    """
    Represent one of the possible labels
    """
    def __init__(self, kernel):
        self.kernel = kernel

        # prediction vector
        self.w = {}

        self.errors = 0

    def train(self, training_list, labels):
        ind = -1
        for x, y_real in zip(training_list, labels):
            ind += 1

            # prediction using the prediction vector
            prediction = 0
            x, v1 = self.kernel.vectorize(x)
            v2 = self.w
            # iterate on the shortest dok
            dok1, dok2 = (v1, v2) if len(v1) < len(v2) else (v2, v1)
            for i in dok1:
                # if both indexes contains non zero elements, sum their product
                if i in dok2:
                    prediction += dok1[i] * dok2[i]

            y_predicted = copysign(1, prediction)

            if y_predicted != y_real:  # wrong prediction
                self.errors += 1
                # update prediction vector
                x, v1 = self.kernel.vectorize(x)
                for _, __ in v1.items():
                    self.w[_] = self.w[_]+y_real*__ if _ in self.w else y_real*__

    def predict_single(self, x):
        x, v1 = self.kernel.vectorize(x)
        v2 = self.w
        score = 0
        # iterate on the shortest dok
        dok1, dok2 = (v1, v2) if len(v1) < len(v2) else (v2, v1)
        for i in dok1:
            # if both indexes contains non zero elements, sum their product
            if i in dok2:
                score += dok1[i] * dok2[i]
        return score
