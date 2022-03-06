import pandas as pd
from source.utils.create_inside_outside_strings import InsideOutside


class PopulateCKYChart:
    def __init__(
        self,
        sentence,
        single_span,
        whole_span,
        verbose,
        likely_constituents,
        dummy_sentence,
    ):
        self.sentence = sentence
        self.likely_constituents = likely_constituents
        self.dummy_sentence = dummy_sentence
        self.sentence_length = len(sentence.split())
        self.verbose = verbose
        self.single_span = single_span
        self.whole_span = whole_span

    def fill_cell(self, span, eval_data):
        cell_score = eval_data.loc[eval_data["span"] == span, "scores"].item()
        return cell_score

    def bracket_crossing(self, span):
        bracket_crossing_list = []
        for low in range(span[0], span[1]):
            for high in range(span[1] + 1, self.sentence_length + 1):
                # Avoid checking for single span and the last span leading to the end of the sentence
                if (high - low) <= 1 or (high - low) == self.sentence_length:
                    continue
                bracket_crossing_list.append((low, high))
        return bracket_crossing_list

    def fill_chart(self):
        span_scores = np.zeros(
            (self.sentence_length + 1, self.sentence_length + 1), dtype=float
        )
        all_spans = helper.TreeHelper(self.sentence).generate_ngrams(
            single_span=self.single_span, whole_span=self.whole_span
        )

        inside_string = []
        outside_string = []

        for span in all_spans:
            inside_outside_spans_dict = InsideOutside(
                sentence=self.sentence
            ).create_inside_outside_matrix(span)
            inside_string.append(inside_outside_spans_dict["inside_string"])
            outside_string.append(
                inside_outside_spans_dict["left_outside_string"].split()[-1]
                + " "
                + "<mask>"
                + " "
                + inside_outside_spans_dict["right_outside_string"].split()[0]
            )

        # TODO: Put comma's under inside string also as it is getting affected due to it's presence
        df = pd.DataFrame(
            {
                "inside_sentence": inside_string,
                "outside_sentence": outside_string,
                "span": [span[0] for span in all_spans],
            }
        )

        stop = helper.get_top_tokens(1000)
        stop.remove("&")
        # Adding certain punctuation
        stop = stop + [",", "-LRB-", "-RRB-", ";"]
        s = df[
            (
                df["inside_sentence"].apply(
                    lambda x: all([item not in stop for item in x.split()])
                )
            )
            & (df["inside_sentence"].str.split().str.len() > 1)
        ]

        df["inside_scores"] = SpanPredict(
            df[["inside_sentence"]], span_method="Inside"
        ).predict_batch(labels=False)
        df["scores"] = scores

        # Single-word consitituents
        df.loc[df["inside_sentence"].str.split().str.len() == 1, "scores"] = 1
        # Whole-sentence constituents
        df.loc[
            df["inside_sentence"].str.split().str.len() == self.sentence_length,
            "scores",
        ] = 1

        # clip lower and upper
        df["scores"] = df["scores"].clip(lower=0.0, upper=1.0)

        assert df["scores"].min() >= 0 and df["scores"].max() <= 1

        for span in all_spans:
            for i in range(0, self.sentence_length):
                for j in range(i + 1, self.sentence_length + 1):
                    if span[0] == (i, j):
                        if self.verbose:
                            pass  # print(i, j, "{:.4f}".format(self.fill_cell(span[0], df)), span[1])
                        span_scores[i, j] = self.fill_cell(span[0], df)
        return self.sentence, span_scores
