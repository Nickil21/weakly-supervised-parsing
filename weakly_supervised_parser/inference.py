import random
import numpy as np
import pandas as pd
from argparse import ArgumentParser

from weakly_supervised_parser.utils.prepare_dataset import DataLoaderHelper
from weakly_supervised_parser.utils.cky_algorithm import get_best_parse
from weakly_supervised_parser.utils.populate_chart import PopulateCKYChart
from weakly_supervised_parser.tree.evaluate import calculate_F1_for_spans, tree_to_spans
from weakly_supervised_parser.tree.helpers import get_constituents, get_distituents
from weakly_supervised_parser.model.trainer import InsideOutsideStringClassifier
from weakly_supervised_parser.settings import PTB_TEST_SENTENCES_WITHOUT_PUNCTUATION_PATH, PTB_TEST_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH
from weakly_supervised_parser.settings import PTB_TRAIN_SENTENCES_WITHOUT_PUNCTUATION_PATH, PTB_TRAIN_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH
from weakly_supervised_parser.settings import INSIDE_BOOTSTRAPPED_DATASET_PATH


class Predictor:

    def __init__(self, sentence):
        self.sentence = sentence
        self.sentence_list = sentence.split()

    def obtain_best_parse(self, model, batch_size, return_df=False):
        unique_tokens_flag, span_scores, df = PopulateCKYChart(sentence=self.sentence).fill_chart(model=model, batches=batch_size)

        if unique_tokens_flag:
            best_parse = "(S " + " ".join(["(S " + item  + ")" for item in self.sentence_list]) + ")"
            print("BEST PARSE", best_parse)
        else:
            span_scores_cky_format = span_scores[:-1, 1:]
            best_parse = get_best_parse(sentence=[self.sentence_list], spans=span_scores_cky_format)
        if return_df:
            return best_parse, df
        return best_parse
        
    
def process_test_sample(index, sentence, gold_file_path, model, batch_size):
    best_parse = Predictor(sentence=sentence).obtain_best_parse(model=model, batch_size=batch_size) 
    gold_standard = DataLoaderHelper(input_file_object=gold_file_path)
    sentence_f1 = calculate_F1_for_spans(tree_to_spans(gold_standard[index]), tree_to_spans(best_parse))
    print("Index: {} <> F1: {:.2f}".format(index, sentence_f1))
    return best_parse

            
def main():
    parser = ArgumentParser(description="Inference Pipeline for the Inside Outside String Classifier", add_help=True)
    
    parser.add_argument("--model_name_or_path", type=str, default="roberta-base",
                       help="Path to the model identifier from huggingface.co/models")

    parser.add_argument("--pre_trained_model_path", type=str, required=True,
                       help="Path to the pretrained model")
    
    parser.add_argument("--seed", type=int, default=None,
                       help="Seed for random shuffling")

    parser.add_argument("--save_path", type=str, required=True,
                       help="Path to save the final trees")

    parser.add_argument("--max_seq_length", default=256, type=int,
                       help="The maximum total input sequence length after tokenization")

    args = parser.parse_args()
    
    inside_model = InsideOutsideStringClassifier(model_name_or_path=args.model_name_or_path, max_seq_length=args.max_seq_length)
    inside_model.load_model(pre_trained_model_path=args.pre_trained_model_path, providers="CUDAExecutionProvider")
    
    with open(args.save_path, "w") as out_file:    
        test_sentences = DataLoaderHelper(input_file_object=PTB_TEST_SENTENCES_WITHOUT_PUNCTUATION_PATH).read_lines()
        test_gold_file_path = PTB_TEST_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH
        for test_index, test_sentence in enumerate(test_sentences):
            best_parse = process_test_sample(test_index, test_sentence, test_gold_file_path, inside_model=inside_model)
            out_file.write(best_parse + "\n")

        
if __name__ == "__main__":
    main()
    
# python weakly_supervised_parser/inference.py --model_name_or_path roberta-base --pre_trained_model_path weakly_supervised_parser/model/TRAINED_MODEL/INSIDE/inside.onnx  --max_seq_length 256  --save_path TEMP/predictions/english/inside_model_predictions.txt
