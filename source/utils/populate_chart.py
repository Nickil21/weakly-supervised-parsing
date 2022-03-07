import pandas as pd
import numpy as np

from source.classifier.model import InsideOutsideStringPredictor
from source.utils.prepare_dataset import NGramify
from source.utils.create_inside_outside_strings import InsideOutside


class PopulateCKYChart:
    def __init__(
        self,
        sentence,
        single_span,
        whole_span,
    ):
        self.sentence = sentence
        self.sentence_length = len(sentence.split())
        self.single_span = single_span
        self.whole_span = whole_span

    def fill_cell(self, span, eval_data):
        cell_score = eval_data.loc[eval_data["span"] == span, "scores"].item()
        return cell_score

    def fill_chart(self):
        span_scores = np.zeros((self.sentence_length + 1, self.sentence_length + 1), dtype=float)
        all_spans = NGramify(self.sentence).generate_ngrams(single_span=self.single_span, whole_span=self.whole_span)

        inside_string = []
        outside_string = []

        for span in all_spans:
            inside_outside_spans_dict = InsideOutside(sentence=self.sentence).create_inside_outside_matrix(span)
            inside_string.append(inside_outside_spans_dict["inside_string"])
            outside_string.append(
                inside_outside_spans_dict["left_outside_string"].split()[-1]
                + " "
                + "<mask>"
                + " "
                + inside_outside_spans_dict["right_outside_string"].split()[0]
            )

        # TODO: Put comma's under inside string also as it is getting affected due to it's presence
        df = pd.DataFrame({"inside_sentence": inside_string, "outside_sentence": outside_string, "span": [span[0] for span in all_spans]})

        df["inside_scores"] = InsideOutsideStringPredictor(eval_dataset=df[["inside_sentence"]], span_method="Inside").predict_batch(inside_model_path="")
        # df["outside_scores"] = InsideOutsideStringPredictor(eval_dataset=df[["outside_sentence"]], span_method="Outside").predict_batch(outside_model_path="")
        df["scores"] = df["inside_scores"].copy()

        # Single-word consitituents
        df.loc[df["inside_sentence"].str.split().str.len() == 1, "scores"] = 1
        # Whole-sentence constituents
        df.loc[df["inside_sentence"].str.split().str.len() == self.sentence_length, "scores"] = 1

        # clip lower and upper
        df["scores"] = df["scores"].clip(lower=0.0, upper=1.0)

        assert df["scores"].min() >= 0 and df["scores"].max() <= 1

        for span in all_spans:
            for i in range(0, self.sentence_length):
                for j in range(i + 1, self.sentence_length + 1):
                    if span[0] == (i, j):
                        span_scores[i, j] = self.fill_cell(span[0], df)
        return span_scores
