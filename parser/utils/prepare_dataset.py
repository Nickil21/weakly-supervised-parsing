import csv
import pandas as pd

from sklearn.model_selection import train_test_split
from parser.utils.process_ptb import punctuation_words, currency_tags_words


class NGramify:
    def __init__(self, sentence):
        self.sentence = sentence.split()
        self.sentence_length = len(self.sentence)
        self.ngrams = []

    def generate_ngrams(self, single_span=False, whole_span=False):
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

    def get_row(self, index):
        for i, line in enumerate(self.read_lines()):
            if i == index:
                return line

    def write_lines(self, keys, values):
        with open(self.output_file_object, "w", newline="\n") as output_file:
            dict_writer = csv.DictWriter(output_file, keys, delimiter="\t")
            dict_writer.writeheader()
            dict_writer.writerows(values)


class PTBDataset:
    def __init__(self, training_data_path):
        self.data = pd.read_csv(training_data_path, sep="\t", header=None, names=["sentence"])
        self.data["sentence"] = self.data

    def preprocess(self):
        filterchars = punctuation_words + currency_tags_words
        filterchars = [char for char in filterchars if char not in list(",;") and char not in "``" and char not in "''"]
        self.data["sentence"] = self.data["sentence"].apply(
            lambda row: " ".join([sentence for sentence in row.split() if sentence not in filterchars])
        )
        return self.data

    def seed_bootstrap_constituent(self):
        constituent_slice_one = self.data["sentence"]
        constituent_samples = pd.DataFrame(dict(sentence=constituent_slice_one, label=1))
        return constituent_samples

    def seed_bootstrap_distituent(self):
        distituent_slice_one = self.data["sentence"].str.split().str[:-1].str.join(" ")
        distituent_slice_two = self.data[self.data["sentence"].str.split().str.len() > 30]["sentence"].str.split().str[:-2].str.join(" ")
        distituent_slice_three = self.data[self.data["sentence"].str.split().str.len() > 40]["sentence"].str.split().str[:-3].str.join(" ")
        distituent_slice_four = self.data[self.data["sentence"].str.split().str.len() > 50]["sentence"].str.split().str[:-4].str.join(" ")
        distituent_slice_five = self.data[self.data["sentence"].str.split().str.len() > 60]["sentence"].str.split().str[:-5].str.join(" ")
        distituent_slice_six = self.data[self.data["sentence"].str.split().str.len() > 70]["sentence"].str.split().str[:-6].str.join(" ")
        distituent_samples = pd.DataFrame(
            dict(
                sentence=pd.concat(
                    [
                        distituent_slice_one,
                        distituent_slice_two,
                        distituent_slice_three,
                        distituent_slice_four,
                        distituent_slice_five,
                        distituent_slice_six,
                    ]
                ),
                label=0,
            )
        )
        return distituent_samples

    def aggregate_samples(self):
        constituent_samples = self.seed_bootstrap_constituent()
        distituent_samples = self.seed_bootstrap_distituent()
        df = pd.concat([constituent_samples, distituent_samples], ignore_index=True).drop_duplicates(subset=["sentence"]).dropna(subset=["sentence"])
        df = df[df["sentence"].str.split().str.len() > 1]
        df["sentence"] = df["sentence"].astype(str)
        train, validation = train_test_split(df.head(1000), test_size=0.2, random_state=42, shuffle=True)
        train.to_csv("source/classifier/datasets/train.csv", index=False)
        validation.to_csv("source/classifier/datasets/validation.csv", index=False)

    def train_validation_split(self):
        pass


if __name__ == "__main__":
    ptb = PTBDataset(training_data_path="./data/PROCESSED/english/ptb-train-sentences-with-punctuation.txt")
    # print(ptb.preprocess())
    print(ptb.aggregate_samples())
