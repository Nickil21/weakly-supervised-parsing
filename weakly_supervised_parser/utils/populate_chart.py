import os
import pandas as pd
import numpy as np

from nltk.corpus import stopwords

from weakly_supervised_parser.utils.prepare_dataset import NGramify
from weakly_supervised_parser.utils.create_inside_outside_strings import InsideOutside
from weakly_supervised_parser.settings import INSIDE_MODEL_PATH
from weakly_supervised_parser.model.span_classifier import InsideOutsideStringPredictor

STOP = set(stopwords.words("english"))
# STOP.update(helper.get_top_tokens(20))

inside_model = InsideOutsideStringPredictor(model_name_or_path="roberta-base", pre_trained_model_path=INSIDE_MODEL_PATH + "inside.onnx")
# outside_model = InsideOutsideStringPredictor(model_name_or_path="roberta-base", pre_trained_model_path=OUTSIDE_MODEL_PATH + "outside.onnx")

    
class PopulateCKYChart:
    def __init__(self, sentence):
        self.sentence = sentence
        self.sentence_length = len(sentence.split())
        self.span_scores = np.zeros((self.sentence_length + 1, self.sentence_length + 1), dtype=float)
        self.all_spans = NGramify(self.sentence).generate_ngrams(single_span=True, whole_span=True)

    def fill_chart(self):
        inside_strings = []
        outside_strings = []

        for span in self.all_spans:
            _, inside_string, outside_string = InsideOutside(sentence=self.sentence).create_inside_outside_matrix(span)
            inside_strings.append(inside_string)
            outside_strings.append(outside_string)

        data = pd.DataFrame({"inside_sentence": inside_strings, "outside_sentence": outside_strings, "span": [span[0] for span in self.all_spans]})
        
        data["inside_scores"] = data["inside_sentence"].apply(inside_model.predict_span)
        data["scores"] = data["inside_scores"].copy()
        
        data.loc[(data['inside_sentence'].str.lower().str.startswith("the")) &
                 (data['inside_sentence'].str.lower().str.split().str.len() == 2) &
                 (~data['inside_sentence'].str.lower().str.split().str[-1].isin(STOP)), 'scores'] = 1

        for span in self.all_spans:
            for i in range(0, self.sentence_length):
                for j in range(i + 1, self.sentence_length + 1):
                    if span[0] == (i, j):
                        self.span_scores[i, j] = data.loc[data["span"] == span[0], "scores"].item()
        return self.span_scores
