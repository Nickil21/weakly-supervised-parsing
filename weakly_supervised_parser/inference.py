from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass

from weakly_supervised_parser.utils.prepare_dataset import DataLoaderHelper
from weakly_supervised_parser.utils.cky_algorithm import get_best_parse
from weakly_supervised_parser.utils.populate_chart import PopulateCKYChart
from weakly_supervised_parser.tree.evaluate import calculate_F1_for_spans, tree_to_spans, evaluate_prediction_file
from weakly_supervised_parser.settings import PTB_TEST_SENTENCES_WITHOUT_PUNCTUATION_PATH, PTB_TEST_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH, PTB_SAVE_TREES

    
class Predictor:

    def __init__(self, sentence):
        self.sentence = sentence
        self.sentence_list = sentence.split()

    def obtain_best_parse(self):
        span_scores = PopulateCKYChart(sentence=self.sentence).fill_chart()
        span_scores_cky_format = span_scores[:-1, 1:]
        best_parse = get_best_parse(sentence=[self.sentence_list], spans=span_scores_cky_format)
        return best_parse

    
def compute_sentence_f1(gold_standard, best_parse):
    return calculate_F1_for_spans(tree_to_spans(gold_standard), tree_to_spans(best_parse))
        
    
def process(test_index, test_sent_without_punct): 
    best_parse = Predictor(sentence=test_sent_without_punct).obtain_best_parse() 
    gold_standard = DataLoaderHelper(input_file_object=PTB_TEST_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH)
    sentence_f1 = compute_sentence_f1(gold_standard[test_index], best_parse)
    print("Index: {} <> F1: {:.2f}".format(test_index, sentence_f1))
    return best_parse
#     out_file.write(best_parse + "\n")
    
    
def mp_handler():
    pool = Pool(processes=16)
    PRED_FILE = PTB_SAVE_TREES + "inside_model_predictions.txt"
    test_sents_without_punct = DataLoaderHelper(input_file_object=PTB_TEST_SENTENCES_WITHOUT_PUNCTUATION_PATH).read_lines()
    with open(PRED_FILE, "w") as out_file:
        for best_parse in pool.starmap(process, list(enumerate(test_sents_without_punct[:100]))):
            f.write(best_parse + "\n")
        
        
if __name__ == "__main__":
    mp_handler()
    
   
    
           
#     for len_limit in [None, 10]:
#         evaluate_prediction_file(gold_file=PTB_TEST_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH, pred_file=PRED_FILE, len_limit=len_limit)
        
        
       