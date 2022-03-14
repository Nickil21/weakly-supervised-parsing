import pandas as pd
import numpy as np

from datasets.utils import set_progress_bar_enabled

from weakly_supervised_parser.utils.prepare_dataset import NGramify
from weakly_supervised_parser.utils.create_inside_outside_strings import InsideOutside
from weakly_supervised_parser.settings import INSIDE_MODEL_PATH
from weakly_supervised_parser.model.span_classifier import InsideOutsideStringPredictor
from weakly_supervised_parser.utils.distant_supervision import RuleBasedHeuristic
from weakly_supervised_parser.utils.prepare_dataset import PTBDataset
from weakly_supervised_parser.settings import PTB_TRAIN_SENTENCES_WITHOUT_PUNCTUATION_PATH

# Disable Dataset.map progress bar
set_progress_bar_enabled(False)

ptb = PTBDataset(training_data_path=PTB_TRAIN_SENTENCES_WITHOUT_PUNCTUATION_PATH)
STOP = [item.lower() for item in RuleBasedHeuristic(corpus=ptb.retrieve_all_sentences()).get_top_tokens(top_most_common_ptb=100)]

inside_model = InsideOutsideStringPredictor(
    model_name_or_path="roberta-base", pre_trained_model_path=INSIDE_MODEL_PATH + "inside.onnx", max_length=200
)

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
        inside_scores = []
        outside_scores = []

        for span in self.all_spans:
            _, inside_string, outside_string = InsideOutside(sentence=self.sentence).create_inside_outside_matrix(span)
            inside_strings.append(inside_string)
            outside_strings.append(outside_string)

        data = pd.DataFrame({"inside_sentence": inside_strings, "outside_sentence": outside_strings, "span": [span[0] for span in self.all_spans]})

        inside_scores.extend(inside_model.predict_span(spans=data.rename(columns={"inside_sentence": "sentence"})[["sentence"]]))
        # outside_scores.extend(outside_model.predict_span(spans=data.rename(columns={"outside_sentence": "sentence"})[["sentence"]]))

        data["inside_scores"] = inside_scores

        data.loc[
            (data["inside_sentence"].str.lower().str.startswith("the"))
            & (data["inside_sentence"].str.lower().str.split().str.len() == 2)
            & (~data["inside_sentence"].str.lower().str.split().str[-1].isin(RuleBasedHeuristic().get_top_tokens())),
            "inside_scores",
        ] = 1

        is_upper_or_title = all([item.istitle() or item.isupper() for item in self.sentence.split()])
        is_stop = any([item for item in self.sentence.split() if item.lower() in STOP])

        flags = is_upper_or_title and not is_stop

        # data["outside_scores"] = outside_scores
        data["scores"] = data["inside_scores"].copy()

        for span in self.all_spans:
            for i in range(0, self.sentence_length):
                for j in range(i + 1, self.sentence_length + 1):
                    if span[0] == (i, j):
                        self.span_scores[i, j] = data.loc[data["span"] == span[0], "scores"].item()
        return flags, self.span_scores
