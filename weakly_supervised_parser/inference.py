from weakly_supervised_parser.utils.prepare_dataset import DataLoaderHelper
from weakly_supervised_parser.utils.cky_algorithm import get_best_parse
from weakly_supervised_parser.utils.populate_chart import PopulateCKYChart
from weakly_supervised_parser.utils.distant_supervision import RuleBasedHeuristic
from weakly_supervised_parser.tree.evaluate import calculate_F1_for_spans, tree_to_spans, evaluate_prediction_file
from weakly_supervised_parser.settings import PTB_TEST_SENTENCES_WITHOUT_PUNCTUATION_PATH, PTB_TEST_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH, PTB_SAVE_TREES

    
class Predictor:

    def __init__(self, sentence):
        self.sentence = sentence
        self.sentence_list = sentence.split()

    def predict(self):
        span_scores = PopulateCKYChart(sentence=self.sentence).fill_chart()
        span_scores_cky_format = span_scores[:-1, 1:]
        best_parse = get_best_parse(sentence=[self.sentence_list], spans=span_scores_cky_format)
        return best_parse


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
        for test_index, test_sent_without_punct in enumerate(test_sents_without_punct):
            out_file.write(obtain_best_parse(test_index, test_sent_without_punct) + "\n")
            if test_index == 100:
                break
        
    for len_limit in [None, 10]:
        evaluate_prediction_file(gold_file=PTB_TEST_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH, pred_file=PRED_FILE, len_limit=len_limit)
    
    

# inside-epochs=00-val_loss=0.1886.ckpt
#
# =====> Evaluation Results <=====
# Length constraint: None
# Micro F1: 49.88, Macro F1: 51.24
# =====> Evaluation Results <=====
# =====> Evaluation Results <=====
# Length constraint: 10
# Micro F1: 55.46, Macro F1: 59.95
# =====> Evaluation Results <=====
