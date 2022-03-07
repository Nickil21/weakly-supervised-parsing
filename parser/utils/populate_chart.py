import pandas as pd
import numpy as np

from parser.train.span_classifier import InsideOutsideStringPredictor
from parser.utils.prepare_dataset import NGramify
from parser.utils.create_inside_outside_strings import InsideOutside


class PopulateCKYChart:
    def __init__(self, sentence):
        self.sentence = sentence
        self.sentence_length = len(sentence.split())
        self.span_scores = np.zeros((self.sentence_length + 1, self.sentence_length + 1), dtype=float)
        self.all_spans = NGramify(self.sentence).generate_ngrams(single_span=True, whole_span=True)

    def fill_cell(self, span, eval_data):
        cell_score = eval_data.loc[eval_data["span"] == span, "scores"].item()
        return cell_score
        
    def sanity_check(self, eval_data):
        eval_data.loc[eval_data["inside_sentence"].str.split().str.len() == 1, "scores"] = 1
        eval_data.loc[eval_data["inside_sentence"].str.split().str.len() == self.sentence_length, "scores"] = 1
        eval_data["scores"] = eval_data["scores"].clip(lower=0.0, upper=1.0)
        return eval_data

    def fill_chart(self, inside_model_path=None, outside_model_path=None):
        inside_strings = []
        outside_strings = []

        for span in self.all_spans:
            _, inside_string, outside_string = InsideOutside(sentence=self.sentence).create_inside_outside_matrix(span)
            inside_strings.append(inside_string)
            outside_strings.append(outside_string)

        data = pd.DataFrame({"inside_sentence": inside_strings, "outside_sentence": outside_strings, "span": [span[0] for span in self.all_spans]})
        data["inside_scores"] = InsideOutsideStringPredictor(eval_dataset=data[["inside_sentence"]], span_method="Inside").predict_batch(inside_model_path=inside_model_path)
        # data["outside_scores"] = InsideOutsideStringPredictor(eval_dataset=data[["outside_sentence"]], span_method="Outside").predict_batch(outside_model_path=outside_model_path)
        data["scores"] = data["inside_scores"].copy()

        data = self.sanity_check(eval_data=data)

        for span in self.all_spans:
            for i in range(0, self.sentence_length):
                for j in range(i + 1, self.sentence_length + 1):
                    if span[0] == (i, j):
                        self.span_scores[i, j] = self.fill_cell(span[0], data)
        return self.span_scores
