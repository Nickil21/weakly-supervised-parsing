import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from weakly_supervised_parser.settings import PTB_TRAIN_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH
from weakly_supervised_parser.settings import PTB_TRAIN_SENTENCES_WITHOUT_PUNCTUATION_PATH
from weakly_supervised_parser.utils.prepare_dataset import DataLoaderHelper
from weakly_supervised_parser.inference import process_test_sample


def prepare_outside_strings(inside_model, upper_threshold, lower_threshold, num_train_rows, seed):
    train_sentences = DataLoaderHelper(input_file_object=PTB_TRAIN_SENTENCES_WITHOUT_PUNCTUATION_PATH).read_lines()
    train_gold_file_path = PTB_TRAIN_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH
    lst = []
    for train_index, train_sentence in enumerate(train_sentences):
        best_parse, df = process_test_sample(train_index, train_sentence, train_gold_file_path, predict_type="inside", model=inside_model, return_df=True)
        
        outside_constituent_samples = pd.DataFrame(dict(sentence=df.loc[df["scores"] > upper_threshold, "outside_sentence"].values,
                                                        label=1)
                                                  )
        
        outside_distituent_samples = pd.DataFrame(dict(sentence=df.loc[df["scores"] < lower_threshold, "outside_sentence"].values, 
                                                       label=0)
                                                 )
        
        lst.append(pd.concat([outside_constituent_samples, outside_distituent_samples]))
        
        if train_index == num_train_rows:
            break
            
    df_outside = pd.concat(lst, ignore_index=True)
    df_outside.drop_duplicates(subset=["sentence"], inplace=True)
    df_outside.reset_index(drop=True, inplace=True)
    train_outside, validation_outside = train_test_split(df_outside, test_size=0.5, random_state=seed, shuffle=True)
    return train_outside, validation_outside


def prepare_data_for_co_training(inside_model, outside_model, upper_threshold, lower_threshold, seed):
     train_sentences = DataLoaderHelper(input_file_object=PTB_TRAIN_SENTENCES_WITHOUT_PUNCTUATION_PATH).read_lines()
     train_gold_file_path = PTB_TRAIN_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH

     for train_index, train_sentence in enumerate(train_sentences):
        _, df_inside = process_test_sample(train_index, train_sentence, train_gold_file_path, predict_type="inside", model=inside_model, return_df=True)
        _, df_outside = process_test_sample(train_index, train_sentence, train_gold_file_path, predict_type="outside", model=outside_model, return_df=True)

        outside_constituent_from_most_confident_inside = df_inside.loc[df_inside["scores"] > upper_threshold, "outside_sentence"].values
        outside_distituent_from_most_confident_inside = df_inside.loc[df_inside["scores"] < lower_threshold, "outside_sentence"].values

        inside_constituent_from_most_confident_outside = df_outside.loc[df_outside["scores"] > upper_threshold, "inside_sentence"].values
        inside_distituent_from_most_confident_outside = df_outside.loc[df_outside["scores"] < lower_threshold, "inside_sentence"].values

        outside_from_most_confident_inside = pd.DataFrame(dict(sentence=outside_constituent_from_most_confident_inside, label=1,
                                                               sentence=outside_distituent_from_most_confident_inside, label=0)
                                                               )

        inside_from_most_confident_outside = pd.DataFrame(dict(sentence=inside_constituent_from_most_confident_outside, label=1,
                                                               sentence=inside_distituent_from_most_confident_outside, label=0))

        return inside_from_most_confident_outside.sample(frac=1., random_state=seed), outside_from_most_confident_inside.sample(frac=1., random_state=seed)
        

class CoTrainingClassifier:
    def __init__(
        self, 
        inside_model, 
        outside_model, 
        pos: int = -1, 
        neg: int = -1, 
        num_iterations: int = 2, 
        pool_of_unlabeled_samples: int = 1000
        ):
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

        # let's fit our final model
        self.inside_model.fit(inside_string[labeled_samples], y[labeled_samples])
        self.outside_model.fit(outside_string[labeled_samples], y[labeled_samples])

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
