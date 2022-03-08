from parser.utils.prepare_dataset import DataLoader
from parser.utils.cky_algorithm import get_best_parse
from parser.utils.populate_chart import PopulateCKYChart
from parser.tree.evaluate import calculate_F1_for_spans, tree_to_spans
from parser.settings import PTB_TEST_SENTENCES_WITHOUT_PUNCTUATION_PATH, PTB_TEST_SENTENCES_WITH_PUNCTUATION_PATH
from parser.settings import PTB_TEST_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH


class Predictor:

    def __init__(self, sentence):
        self.sentence = sentence
        self.sentence_list = sentence.split()

    def predict(self):
        span_scores = PopulateCKYChart(sentence=self.sentence).fill_chart()
        span_scores_cky_format = span_scores[:-1, 1:]
        best_parse = get_best_parse(sentence=[self.sentence_list], spans=span_scores_cky_format)
        return best_parse
        

test_sentences_with_punctuation = DataLoader(input_file_object=PTB_TEST_SENTENCES_WITH_PUNCTUATION_PATH).read_lines()
test_sentences_without_punctuation = DataLoader(input_file_object=PTB_TEST_SENTENCES_WITHOUT_PUNCTUATION_PATH).read_lines()

for (test_index, (test_sentence_with_punctuation, test_sentence_without_punctuation)) in enumerate(zip(test_sentences_with_punctuation, test_sentences_without_punctuation)):
    gold_standard = DataLoader(input_file_object=PTB_TEST_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH)[test_index]
    best_parse = Predictor(sentence=test_sentence_without_punctuation).predict()
    f1 = calculate_F1_for_spans(tree_to_spans(gold_standard), tree_to_spans(best_parse))
    print(f1)
    break
    
