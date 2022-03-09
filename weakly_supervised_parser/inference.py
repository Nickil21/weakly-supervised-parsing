import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from weakly_supervised_parser.utils.prepare_dataset import DataLoaderHelper
from weakly_supervised_parser.utils.cky_algorithm import get_best_parse
from weakly_supervised_parser.utils.populate_chart import PopulateCKYChart
from weakly_supervised_parser.tree.evaluate import calculate_F1_for_spans, tree_to_spans, evaluate_prediction_file
from weakly_supervised_parser.settings import PTB_TEST_SENTENCES_WITHOUT_PUNCTUATION_PATH, PTB_TEST_SENTENCES_WITH_PUNCTUATION_PATH
from weakly_supervised_parser.settings import PTB_TEST_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH
from weakly_supervised_parser.settings import INSIDE_MODEL_PATH
from weakly_supervised_parser.settings import PTB_SAVE_TREES


class Predictor:

    def __init__(self, sentence):
        self.sentence = sentence
        self.sentence_list = sentence.split()

    def predict(self):
        span_scores = PopulateCKYChart(sentence=self.sentence).fill_chart()
        span_scores_cky_format = span_scores[:-1, 1:]
        best_parse = get_best_parse(sentence=[self.sentence_list], spans=span_scores_cky_format)
        return best_parse


# test_sents_with_punct = DataLoaderHelper(input_file_object=PTB_TEST_SENTENCES_WITH_PUNCTUATION_PATH).read_lines()
test_sents_without_punct = DataLoaderHelper(input_file_object=PTB_TEST_SENTENCES_WITHOUT_PUNCTUATION_PATH).read_lines()


def obtain_best_parse(test_index, test_sent_without_punct):
    gold_standard = DataLoaderHelper(input_file_object=PTB_TEST_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH)[test_index]
    best_parse = Predictor(sentence=test_sent_without_punct).predict()
    f1 = calculate_F1_for_spans(tree_to_spans(gold_standard), tree_to_spans(best_parse))
    print("F1: {:.2f}".format(f1))
    return best_parse
 
    
if __name__ == "__main__":
    
    PRED_FILE = PTB_SAVE_TREES + "inside_model_predictions.txt"
    
    with open(PRED_FILE, "w") as out_file:
        for test_index, test_sent_without_punct in enumerate(test_sents_without_punct[:100]):
            out_file.write(obtain_best_parse(test_index, test_sent_without_punct) + "\n")
        
    for len_limit in [None, 10]:
        evaluate_prediction_file(gold_file=PTB_TEST_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH, pred_file=PRED_FILE, len_limit=len_limit)


# [46.15384615384615, 33.333333333333336, 47.99999999999999, 25.0, 53.333333333333336, 36.66666666666667, 47.61904761904762, 64.86486486486486, 48.48484848484848, 58.82352941176471]
    
