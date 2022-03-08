import csv
import pandas as pd

from sklearn.model_selection import train_test_split
from weakly_supervised_parser.utils.process_ptb import punctuation_words, currency_tags_words
from weakly_supervised_parser.utils.distant_supervision import RuleBasedHeuristic
from weakly_supervised_parser.settings import PTB_TRAIN_SENTENCES_WITH_PUNCTUATION_PATH
from weakly_supervised_parser.settings import INSIDE_BOOTSTRAPPED_DATASET_PATH


class NGramify:
    def __init__(self, sentence):
        self.sentence = sentence.split()
        self.sentence_length = len(self.sentence)
        self.ngrams = []

    def generate_ngrams(self, single_span=True, whole_span=True):
        # number of substrings possible is N*(N+1)/2
        # exclude substring or spans of length 1 and length N
        if single_span:
            start = 1
        else:
            start = 2
        if whole_span:
            end = self.sentence_length + 1
        else:
            end = self.sentence_length
        for n in range(start, end):
            for i in range(self.sentence_length - n + 1):
                self.ngrams.append(((i, i + n), self.sentence[i : i + n]))
        return self.ngrams

    def generate_all_possible_spans(self):
        for n in range(2, self.sentence_length):
            for i in range(self.sentence_length - n + 1):
                if i > 0 and (i + n) < self.sentence_length:
                    self.ngrams.append(
                        (
                            (i, i + n),
                            " ".join(self.sentence[i : i + n]),
                            " ".join(self.sentence[0:i])
                            + " ("
                            + " ".join(self.sentence[i : i + n])
                            + ") "
                            + " ".join(self.sentence[i + n : self.sentence_length]),
                        )
                    )
                elif i == 0:
                    self.ngrams.append(
                        (
                            (i, i + n),
                            " ".join(self.sentence[i : i + n]),
                            "(" + " ".join(self.sentence[i : i + n]) + ") " + " ".join(self.sentence[i + n : self.sentence_length]),
                        )
                    )
                elif (i + n) == self.sentence_length:
                    self.ngrams.append(
                        (
                            (i, i + n),
                            " ".join(self.sentence[i : i + n]),
                            " ".join(self.sentence[0:i]) + " (" + " ".join(self.sentence[i : i + n]) + ")",
                        )
                    )
        return self.ngrams


class DataLoader:
    def __init__(self, input_file_object=None, output_file_object=None):
        self.input_file_object = input_file_object
        self.output_file_object = output_file_object

    def read_lines(self):
        with open(self.input_file_object, "r") as f:
            lines = f.read().splitlines()
        return lines

    def __getitem__(self, index):
        return self.read_lines()[index]

    def write_lines(self, keys, values):
        with open(self.output_file_object, "w", newline="\n") as output_file:
            dict_writer = csv.DictWriter(output_file, keys, delimiter="\t")
            dict_writer.writeheader()
            dict_writer.writerows(values)


class PTBDataset:
    def __init__(self, training_data_path):
        self.data = pd.read_csv(training_data_path, sep="\t", header=None, names=["sentence"])
        self.data["sentence"] = self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data["sentence"].loc[index]

    def retrieve_all_sentences(self):
        return self.data["sentence"].tolist()

    def preprocess(self):
        filterchars = punctuation_words + currency_tags_words
        filterchars = [char for char in filterchars if char not in list(",;") and char not in "``" and char not in "''"]
        self.data["sentence"] = self.data["sentence"].apply(lambda row: " ".join([sentence for sentence in row.split() if sentence not in filterchars]))
        return self.data

    def seed_bootstrap_constituent(self):
        whole_span_slice = self.data["sentence"]
        func = lambda x: RuleBasedHeuristic().add_contiguous_titlecase_words(row=[(index, character) for index, character in enumerate(x) if character.istitle() or "'" in character])
        titlecase_matches = [item for sublist in self.data['sentence'].str.split().apply(func).tolist() for item in sublist if len(item.split()) > 1]
        titlecase_matches_df = pd.Series(titlecase_matches)
        most_frequent_start_token = RuleBasedHeuristic(corpus=self.retrieve_all_sentences()).augment_using_most_frequent_starting_token()
        most_frequent_start_token_df = titlecase_matches_df[titlecase_matches_df.str.startswith(most_frequent_start_token)].str.lower()
        constituent_samples = pd.DataFrame(dict(sentence=pd.concat([whole_span_slice, titlecase_matches_df, most_frequent_start_token_df]), label=1))
        return constituent_samples

    def seed_bootstrap_distituent(self):
        last_but_one_slice = self.data["sentence"].str.split().str[:-1].str.join(" ")
        last_but_two_slice = self.data[self.data["sentence"].str.split().str.len() > 30]["sentence"].str.split().str[:-2].str.join(" ")
        last_but_three_slice = self.data[self.data["sentence"].str.split().str.len() > 40]["sentence"].str.split().str[:-3].str.join(" ")
        last_but_four_slice = self.data[self.data["sentence"].str.split().str.len() > 50]["sentence"].str.split().str[:-4].str.join(" ")
        last_but_five_slice = self.data[self.data["sentence"].str.split().str.len() > 60]["sentence"].str.split().str[:-5].str.join(" ")
        last_but_six_slice = self.data[self.data["sentence"].str.split().str.len() > 70]["sentence"].str.split().str[:-6].str.join(" ")
        distituent_samples = pd.DataFrame(
            dict(
                sentence=pd.concat(
                    [
                        last_but_one_slice,
                        last_but_two_slice,
                        last_but_three_slice,
                        last_but_four_slice,
                        last_but_five_slice,
                        last_but_six_slice,
                    ]
                ),
                label=0,
            )
        )
        return distituent_samples

    def train_validation_split(self, test_size=0.2, shuffle=True, seed=42):
        self.preprocess()
        bootstrap_constituent_samples = self.seed_bootstrap_constituent()
        bootstrap_distituent_samples = self.seed_bootstrap_distituent()
        df = pd.concat([bootstrap_constituent_samples, bootstrap_distituent_samples], ignore_index=True)
        df = df.drop_duplicates(subset=["sentence"]).dropna(subset=["sentence"])
        df = df[df["sentence"].str.split().str.len() > 1]
        train, validation = train_test_split(df, test_size=test_size, random_state=seed, shuffle=shuffle)
        train = train.head(8000)
        validation = validation.head(2000)
        train.to_csv(INSIDE_BOOTSTRAPPED_DATASET_PATH + "train.csv", index=False)
        validation.to_csv(INSIDE_BOOTSTRAPPED_DATASET_PATH + "validation.csv", index=False)


if __name__ == "__main__":
    ptb = PTBDataset(training_data_path=PTB_TRAIN_SENTENCES_WITH_PUNCTUATION_PATH)
    print(ptb.train_validation_split())