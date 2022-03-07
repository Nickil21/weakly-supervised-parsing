from parser.utils.prepare_dataset import DataLoader
from parser.utils.cky_algorithm import get_best_parse
from parser.utils.populate_chart import PopulateCKYChart


class Predictor:

    def __init__(self, sentence):
        self.sentence = sentence
        self.sentence_list = sentence.split()

    def predict(self, single_span, whole_span):
        span_scores = PopulateCKYChart(sentence=self.sentence, single_span=single_span, whole_span=whole_span).fill_chart()
        span_scores_cky_format = span_scores[:-1, 1:]
        best_parse = get_best_parse(sentence=[self.sentence_list], spans=span_scores_cky_format)
        return best_parse
        