

   
from parser.utils.align_ptb import AlignPTBYoonKimFormat
from parser.utils.prepare_dataset import DataLoader
from parser.utils.cky_algorithm import get_best_parse
from parser.utils.populate_chart import PopulateCKYChart


from parser.settings import PTB_TRAIN_GOLD_PATH, PTB_VALID_GOLD_PATH, PTB_TEST_GOLD_PATH
from parser.settings import PTB_TRAIN_GOLD_ALIGNED_PATH, PTB_VALID_GOLD_ALIGNED_PATH, PTB_TEST_GOLD_ALIGNED_PATH
from parser.settings import YOON_KIM_TRAIN_GOLD_PATH, YOON_KIM_VALID_GOLD_PATH, YOON_KIM_TEST_GOLD_PATH

# Align PTB with YK
AlignPTBYoonKimFormat(ptb_data_path=PTB_TRAIN_GOLD_PATH, yk_data_path=YOON_KIM_TRAIN_GOLD_PATH).row_mapper(save_data_path=PTB_TRAIN_GOLD_ALIGNED_PATH)
AlignPTBYoonKimFormat(ptb_data_path=PTB_VALID_GOLD_PATH, yk_data_path=YOON_KIM_VALID_GOLD_PATH).row_mapper(save_data_path=PTB_VALID_GOLD_ALIGNED_PATH)
AlignPTBYoonKimFormat(ptb_data_path=PTB_TEST_GOLD_PATH, yk_data_path=YOON_KIM_TEST_GOLD_PATH).row_mapper(save_data_path=PTB_TEST_GOLD_ALIGNED_PATH)



class Predictor:

    def __init__(self, sentence):
        self.sentence = sentence
        self.sentence_list = sentence.split()

    def predict(self, single_span, whole_span):
        span_scores = PopulateCKYChart(sentence=self.sentence, single_span=single_span, whole_span=whole_span).fill_chart()
        span_scores_cky_format = span_scores[:-1, 1:]
        best_parse = get_best_parse(sentence=[self.sentence_list], spans=span_scores_cky_format)
        return best_parse
        