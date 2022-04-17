import pandas as pd
import numpy as np

from datasets.utils import set_progress_bar_enabled

from weakly_supervised_parser.utils.prepare_dataset import NGramify
from weakly_supervised_parser.utils.create_inside_outside_strings import InsideOutside
from weakly_supervised_parser.utils.cky_algorithm import get_best_parse
from weakly_supervised_parser.utils.distant_supervision import RuleBasedHeuristic
from weakly_supervised_parser.utils.prepare_dataset import PTBDataset
from weakly_supervised_parser.settings import PTB_TRAIN_SENTENCES_WITHOUT_PUNCTUATION_PATH

# Disable Dataset.map progress bar
set_progress_bar_enabled(False)

# ptb = PTBDataset(data_path=PTB_TRAIN_SENTENCES_WITHOUT_PUNCTUATION_PATH)
# ptb_top_100_common = [item.lower() for item in RuleBasedHeuristic(corpus=ptb.retrieve_all_sentences()).get_top_tokens(top_most_common_ptb=100)]
ptb_top_100_common = ['this', 'myself', 'shouldn', 'not', 'analysts', 'same', 'mightn', 'we', 'american', 'the', 'another', 'until', "aren't", 'when', 'if', 'am', 'over', 'ma', 'as', 'of', 'with', 'even', 'couldn', 'not', "needn't", 'where', 'there', 'isn', 'however', 'my', 'sales', 'here', 'at', 'yours', 'into', 'wouldn', 'officials', 'no', "hasn't", 'to', 'wasn', 'any', 'ours', 'out', 'each', "wasn't", 'is', 'and', 'me', 'off', 'once', "it's", 'they', 'most', 'also', 'through', 'hasn', 'our', 'or', 'after', "weren't", 'about', 'mr.', 'first', 'haven', 'needn', 'have', "isn't", 'now', "didn't", 'on', 'theirs', 'these', 'before', 'there', 'was', 'which', 'those', 'having', 'do', 'most', 'own', 'among', 'because', 'for', "should've", "shan't", 'so', 'being', 'few', 'too', 'to', 'at', 'people', 'her', 'meanwhile', 'both', 'down', 'doesn', 'below', 'mustn', 'an', 'two', 'more', 'japanese', 'ford', "you'd", 'about', 'but', 'doing', 'itself', 've', 'under', 'what', 'again', 'then', 'your', 'himself', 'now', 'against', 'just', 'does', 'net', "couldn't", 'that', 'he', 'revenue', 'because', 'yesterday', 'them', 'i', 'their', 'all', 'under', 'up', "haven't", 'while', "won't", 'it', 'more', 'it', 'ain', 'him', 'still', 'a', 'he', 'despite', 'should', 'during', 'nor', "shouldn't", 'such', "doesn't", 'are', "that'll", 'since', 'yourselves', 'such', 'those', 'after', 'weren', "you're", 'd', 'like', 'did', 'hadn', 'themselves', 'its', 'but', 'been', 's', "don't", 'these', 'they', 'this', 'his', "mightn't", 'moreover', 'how', 'new', 'above', 'ourselves', 'so', 'why', 'between', 'their', 'general', "wouldn't", 'who', 'i', 'in', 'don', 'shan', 'u.s.', 'ibm', 'separately', 'had', 'you', 'federal', 'if', 'our', 'and', 'only', 'y', 'many', 'one', 'no', 'though', 'won', 'last', 'from', 'each', 'traders', 'john', 'further', 'hers', 'both', "you've", "you'll", 'that', 'all', 'its', 'only', 'here', 'according', "mustn't", 'while', 'in', 'what', 'didn', 'when', 'some', 'on', 'can', 'yourself', 'herself', 'than', 'with', 'has', 'she', 'during', 'will', 'of', 'thus', 'you', 'very', 'o', 'investors', 'a', 'ms.', 'japan', 'were', 'the', 'we', 'm', 'as', 'll', 'be', 'by', 'other', 'yet', 'whom', 'some', 'indeed', 'other', "she's", "hadn't", 'by', 'earlier', 'for', 'instead', 'she', 'an', 't', 're', 'his', 'then', 'aren', 'although']
# ptb_most_common_first_token = RuleBasedHeuristic(corpus=ptb.retrieve_all_sentences()).augment_using_most_frequent_starting_token(N=1)[0][0].lower()
ptb_most_common_first_token = "the"


class PopulateCKYChart:
    def __init__(self, sentence):
        self.sentence = sentence
        self.sentence_list = sentence.split()
        self.sentence_length = len(sentence.split())
        self.span_scores = np.zeros((self.sentence_length + 1, self.sentence_length + 1), dtype=float)
        self.all_spans = NGramify(self.sentence).generate_ngrams(single_span=True, whole_span=True)

    def compute_scores(self, model, predict_type, scale_axis, predict_batch_size, chunks=128):
        inside_strings = []
        outside_strings = []
        inside_scores = []
        outside_scores = []

        for span in self.all_spans:
            _, inside_string, outside_string = InsideOutside(sentence=self.sentence).create_inside_outside_matrix(span)
            inside_strings.append(inside_string)
            outside_strings.append(outside_string)

        data = pd.DataFrame({"inside_sentence": inside_strings, "outside_sentence": outside_strings, "span": [span[0] for span in self.all_spans]})

        if predict_type == "inside":
            
            if data.shape[0] > chunks:
                data_chunks = np.array_split(data, data.shape[0] // chunks)
                for data_chunk in data_chunks:
                    inside_scores.extend(model.predict_proba(spans=data_chunk.rename(columns={"inside_sentence": "sentence"})[["sentence"]],
                                                             scale_axis=scale_axis,
                                                             predict_batch_size=predict_batch_size)[:, 1])
            else:
                inside_scores.extend(model.predict_proba(spans=data.rename(columns={"inside_sentence": "sentence"})[["sentence"]],
                                                         scale_axis=scale_axis,
                                                         predict_batch_size=predict_batch_size)[:, 1])

            data["inside_scores"] = inside_scores
            data.loc[
                (data["inside_sentence"].str.lower().str.startswith(ptb_most_common_first_token))
                & (data["inside_sentence"].str.lower().str.split().str.len() == 2)
                & (~data["inside_sentence"].str.lower().str.split().str[-1].isin(RuleBasedHeuristic().get_top_tokens())),
                "inside_scores",
            ] = 1

            is_upper_or_title = all([item.istitle() or item.isupper() for item in self.sentence.split()])
            is_stop = any([item for item in self.sentence.split() if item.lower() in ptb_top_100_common])

            flags = is_upper_or_title and not is_stop
            
            data["scores"] = data["inside_scores"]

        elif predict_type == "outside":
            outside_scores.extend(model.predict_proba(spans=data.rename(columns={"outside_sentence": "sentence"})[["sentence"]],
                                                      scale_axis=scale_axis,
                                                      predict_batch_size=predict_batch_size)[:, 1])
            data["outside_scores"] = outside_scores
            flags = False
            data["scores"] = data["outside_scores"]

        return flags, data

    def fill_chart(self, model, predict_type, scale_axis, predict_batch_size, data=None):
        if data is None:
            flags, data = self.compute_scores(model, predict_type, scale_axis, predict_batch_size)
        for span in self.all_spans:
            for i in range(0, self.sentence_length):
                for j in range(i + 1, self.sentence_length + 1):
                    if span[0] == (i, j):
                        self.span_scores[i, j] = data.loc[data["span"] == span[0], "scores"].item()
        return flags, self.span_scores, data

    def best_parse_tree(self, span_scores):
        span_scores_cky_format = span_scores[:-1, 1:]
        best_parse = get_best_parse(sentence=[self.sentence_list], spans=span_scores_cky_format)
        return best_parse
