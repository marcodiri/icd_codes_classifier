from itertools import accumulate
from math import copysign


class VotedPerceptron:
    """
    Represent one of the possible labels
    """
    def __init__(self, kernel):
        self.kernel = kernel

        # prediction vectors list
        self.w = [{}]

        # prediction vector votes generated during training
        self.weights = []

    def train(self, training_list, labels):
        # initialize structures
        self.current_weight = 0
        ind = -1
        for x, y_real in zip(training_list, labels):
            ind += 1

            # prediction using the prediction vector
            prediction = 0
            x, v1 = self.kernel.vectorize(x)
            v2 = self.w[-1].copy()
            # iterate on the shortest dok
            dok1, dok2 = (v1, v2) if len(v1) < len(v2) else (v2, v1)
            for i in dok1:
                # if both indexes contains non zero elements, sum their product
                if i in dok2:
                    prediction += dok1[i] * dok2[i]

            y_predicted = copysign(1, prediction)

            if y_predicted == y_real:  # correct prediction
                self.current_weight += 1
            else:  # wrong prediction
                # update prediction vector
                x, v1 = self.kernel.vectorize(x)
                for _, __ in v1.items():
                    if _ in v2:  # the value is != 0 in both vectors
                        new_value = v2[_] + y_real * __
                        if new_value == 0:  # if the new value is 0, remove the key
                            v2.pop(_)
                        else:
                            v2[_] = new_value
                    else:
                        v2[_] = y_real * __
                self.w.append(v2)
                # save weight
                self.weights.append(self.current_weight)
                self.current_weight = 1  # reset weight

        # training complete
        # save the last weight
        self.weights.append(self.current_weight)

    def predict_single(self, x):
        x, v1 = self.kernel.vectorize(x)
        v2 = self.w[-1]
        score = 0
        # iterate on the shortest dok
        dok1, dok2 = (v1, v2) if len(v1) < len(v2) else (v2, v1)
        for i in dok1:
            # if both indexes contains non zero elements, sum their product
            if i in dok2:
                score += dok1[i] * dok2[i]
        return score

    def predict_voted(self, x):
        x, v1 = self.kernel.vectorize(x)
        score = 0
        # TODO: to be implemented
        return score
