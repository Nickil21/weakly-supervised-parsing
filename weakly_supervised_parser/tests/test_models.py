import numpy as np

from weakly_supervised_parser.utils.prepare_dataset import PTBDataset, DataLoaderHelper
from weakly_supervised_parser.inference import Predictor
from weakly_supervised_parser.tree.evaluate import calculate_F1_for_spans, tree_to_spans
from weakly_supervised_parser.model.span_classifier import InsideOutsideStringPredictor
from weakly_supervised_parser.settings import PTB_VALID_SENTENCES_WITHOUT_PUNCTUATION_PATH, PTB_VALID_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH
from weakly_supervised_parser.settings import INSIDE_MODEL_PATH


validation_sentences = PTBDataset(PTB_VALID_SENTENCES_WITHOUT_PUNCTUATION_PATH).retrieve_all_sentences(N=10)

inside_model = InsideOutsideStringPredictor(model_name_or_path="roberta-base", 
                                            pre_trained_model_path=INSIDE_MODEL_PATH + "inside_model.onnx",
                                            max_length=256)

def test_inside_model():
    sentences_f1 = []
    for index, validation_sentence in enumerate(validation_sentences):
        best_parse = Predictor(sentence=validation_sentence).obtain_best_parse(inside_model=inside_model) 
        gold_standard = DataLoaderHelper(input_file_object=PTB_VALID_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH)
        sentence_f1 = calculate_F1_for_spans(tree_to_spans(gold_standard[index]), tree_to_spans(best_parse))
        sentences_f1.append(sentence_f1)
    assert np.mean(sentences_f1) > 50
