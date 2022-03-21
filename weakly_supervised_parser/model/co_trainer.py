import random
import numpy as np


class CoTrainingClassifier:
    def __init__(self, inside_model, outside_model, pos=-1, neg=-1, num_iterations=5, pool_of_unlabeled_samples=75):
        self.inside_model = inside_model
        self.outside_model = outside_model

        # if they only specify one of neg or pos, throw an exception
        if (pos == -1 and neg != -1) or (pos != -1 and neg == -1):
            raise ValueError("Current implementation supports either both p and n being specified, or neither")

        self.pos_ = pos
        self.neg_ = neg
        self.num_iterations = num_iterations
        self.pool_of_unlabeled_samples = pool_of_unlabeled_samples

        random.seed()

    def fit(self, inside_string, outside_string, y):
        # we need y to be a numpy array so we can do more complex slicing
        y = np.asarray(y)

        # set the n and p parameters if we need to
        if self.pos_ == -1 and self.neg_ == -1:
            num_pos = sum(1 for y_i in y if y_i == 1)
            num_neg = sum(1 for y_i in y if y_i == 0)

            neg_pos_ratio = num_neg / float(num_pos)

            if neg_pos_ratio > 1:
                self.pos_ = 1
                self.neg_ = round(self.pos_ * neg_pos_ratio)

            else:
                self.neg_ = 1
                self.pos_ = round(self.neg_ / neg_pos_ratio)

        assert self.pos_ > 0 and self.neg_ > 0 and self.num_iterations > 0 and self.pool_of_unlabeled_samples > 0

        # the set of unlabeled samples
        unlabeled_samples = [i for i, y_i in enumerate(y) if y_i == -1]

        # we randomize here, and then just take from the back so we don't have to sample every time
        random.shuffle(unlabeled_samples)

        # this is U' in paper
        U_ = unlabeled_samples[-min(len(unlabeled_samples), self.pool_of_unlabeled_samples) :]

        # the samples that are initially labeled
        labeled_samples = [i for i, y_i in enumerate(y) if y_i != -1]

        # remove the samples in U_ from unlabeled samples
        unlabeled_samples = unlabeled_samples[: -len(U_)]

        it = 0  # number of cotraining iterations we've done so far

        # loop until we have assigned labels to everything in unlabeled samples or we hit our iteration break condition
        while it != self.num_iterations and unlabeled_samples:
            it += 1

            self.inside_model.fit(inside_string[labeled_samples], y[labeled_samples])
            self.outside_model.fit(outside_string[labeled_samples], y[labeled_samples])

            inside_prob = self.inside_model.predict_proba(inside_string[U_])
            outside_prob = self.outside_model.predict_proba(outside_string[U_])

            n, p = [], []

            for i in (inside_prob[:, 0].argsort())[-self.neg_ :]:
                if inside_prob[i, 0] > 0.5:
                    n.append(i)
            for i in (inside_prob[:, 1].argsort())[-self.pos_ :]:
                if inside_prob[i, 1] > 0.5:
                    p.append(i)

            for i in (outside_prob[:, 0].argsort())[-self.neg_ :]:
                if outside_prob[i, 0] > 0.5:
                    n.append(i)
            for i in (outside_prob[:, 1].argsort())[-self.pos_ :]:
                if outside_prob[i, 1] > 0.5:
                    p.append(i)

            # label the samples and remove thes newly added samples from U_
            y[[U_[x] for x in p]] = 1
            y[[U_[x] for x in n]] = 0

            labeled_samples.extend([U_[x] for x in p])
            labeled_samples.extend([U_[x] for x in n])

            U_ = [elem for elem in U_ if not (elem in p or elem in n)]

            # add new elements to U_
            add_counter = 0  # number we have added from unlabeled samples to U_
            num_to_add = len(p) + len(n)
            while add_counter != num_to_add and unlabeled_samples:
                add_counter += 1
                U_.append(unlabeled_samples.pop())

            # TODO: Handle the case where the classifiers fail to agree on any of the samples (i.e. both n and p are empty)

        # let's fit our final model
        self.inside_model.fit(inside_string[labeled_samples], y[labeled_samples])
        self.outside_model.fit(outside_string[labeled_samples], y[labeled_samples])

    # TODO: Move this outside of the class into a util file.
    def supports_proba(self, clf, x):
        """Checks if a given classifier supports the 'predict_proba' method, given a single vector x"""
        try:
            clf.predict_proba([x])
            return True
        except:
            return False

    def predict(self, inside_strings, outside_strings):
        inside_preds = self.inside_model.predict(inside_strings)
        outside_preds = self.outside_model.predict(outside_strings)

        proba_supported = self.supports_proba(self.inside_model, inside_strings[0]) and self.supports_proba(self.outside_model, outside_strings[0])

        # fill y_pred with -1 so we can identify the samples in which the classifiers failed to agree
        y_pred = np.asarray([-1] * inside_strings.shape[0])

        for i, (inside_pred, outside_pred) in enumerate(zip(inside_preds, outside_preds)):
            if inside_pred == outside_pred:
                y_pred[i] = inside_pred
            elif proba_supported:
                inside_probs = self.inside_model.predict_proba([inside_strings[i]])[0]
                outside_probs = self.outside_model.predict_proba([outside_strings[i]])[0]
                sum_y_probs = [prob1 + prob2 for (prob1, prob2) in zip(inside_probs, outside_probs)]
                max_sum_prob = max(sum_y_probs)
                y_pred[i] = sum_y_probs.index(max_sum_prob)

            else:
                # the classifiers disagree and don't support probability, so we guess
                y_pred[i] = random.randint(0, 1)

        # check that we did everything right
        assert not (-1 in y_pred)

        return y_pred

    def predict_proba(self, inside_strings, outside_strings):
        """Predict the probability of the samples belonging to each class."""
        y_proba = np.full((inside_strings.shape[0], 2), -1, np.float)

        inside_proba = self.inside_model.predict_proba(inside_strings)
        outside_proba = self.outside_model.predict_proba(outside_strings)

        for i, (y1_i_dist, y2_i_dist) in enumerate(zip(inside_proba, outside_proba)):
            y_proba[i][0] = (y1_i_dist[0] + y2_i_dist[0]) / 2
            y_proba[i][1] = (y1_i_dist[1] + y2_i_dist[1]) / 2

        _epsilon = 0.0001
        assert all(abs(sum(y_dist) - 1) <= _epsilon for y_dist in y_proba)
        return y_proba
