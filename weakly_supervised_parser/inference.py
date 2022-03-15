import random
import pandas as pd
from argparse import ArgumentParser

from weakly_supervised_parser.utils.prepare_dataset import DataLoaderHelper
from weakly_supervised_parser.utils.cky_algorithm import get_best_parse
from weakly_supervised_parser.utils.populate_chart import PopulateCKYChart
from weakly_supervised_parser.tree.evaluate import calculate_F1_for_spans, tree_to_spans
from weakly_supervised_parser.tree.helpers import get_constituents, get_distituents
from weakly_supervised_parser.model.span_classifier import InsideOutsideStringPredictor
from weakly_supervised_parser.settings import PTB_TEST_SENTENCES_WITHOUT_PUNCTUATION_PATH, PTB_TEST_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH
from weakly_supervised_parser.settings import PTB_TRAIN_SENTENCES_WITHOUT_PUNCTUATION_PATH, PTB_TRAIN_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH
from weakly_supervised_parser.settings import INSIDE_BOOTSTRAPPED_DATASET_PATH


class Predictor:

    def __init__(self, sentence):
        self.sentence = sentence
        self.sentence_list = sentence.split()

    def obtain_best_parse(self, inside_model, outside_model, return_df=False):
        unique_tokens_flag, span_scores, df = PopulateCKYChart(sentence=self.sentence).fill_chart(inside_model=inside_model, outside_model=outside_model)
        if unique_tokens_flag:
            best_parse = "(S " + " ".join(["(S " + item  + ")" for item in self.sentence_list]) + ")"
            print("BEST PARSE", best_parse)
        else:
            span_scores_cky_format = span_scores[:-1, 1:]
            best_parse = get_best_parse(sentence=[self.sentence_list], spans=span_scores_cky_format)
        if return_df:
            return best_parse, df
        return best_parse
    
    def bootstrap_constituents_and_distituents(self, seed):
        train_best_parse, train_df = self.obtain_best_parse(return_df=True)
        train_df.rename(columns={"inside_sentence": "sentence"}, inplace=True)
        train_pseudo_constituents = train_df[train_df["sentence"].isin(get_constituents(train_best_parse))].copy()
        train_pseudo_constituents["label"] = 1
        train_pseudo_distituents = train_df[train_df["sentence"].isin(get_distituents(train_best_parse))].copy()
        train_pseudo_distituents["label"] = 0
        train_pseudo_constituents_pseudo_distituents = pd.concat([train_pseudo_constituents, train_pseudo_distituents])
        return train_pseudo_constituents_pseudo_distituents[["sentence", "label"]].sample(frac=1., random_state=seed)
        
        
    
def process_test_sample(index, sentence, gold_file_path, inside_model=None, outside_model=None):
    best_parse = Predictor(sentence=sentence).obtain_best_parse(inside_model=inside_model, outside_model=outside_model) 
    gold_standard = DataLoaderHelper(input_file_object=gold_file_path)
    sentence_f1 = calculate_F1_for_spans(tree_to_spans(gold_standard[index]), tree_to_spans(best_parse))
    print("Index: {} <> F1: {:.2f}".format(index, sentence_f1))
    return best_parse


def process_train_sample(sentence, seed):
    return Predictor(sentence=sentence).bootstrap_constituents_and_distituents(seed)

            
def main():
    parser = ArgumentParser(description="Inference Pipeline for the Inside Outside String Classifier", add_help=True)
    
    parser.add_argument("--model_name_or_path", type=str, default="roberta-base",
                       help="Path to the model identifier from huggingface.co/models")

    parser.add_argument("--pre_trained_model_path", type=str, required=True,
                       help="Path to the pretrained model")

    group = parser.add_mutually_exclusive_group(required=True)
    
    group.add_argument("--predict_on_train", action='store_true',
                        help="Whether to predict on Train or not")
    
    group.add_argument("--predict_on_test", action='store_true',
                        help="Whether to predict on Test or not")
    
    parser.add_argument("--randomN", type=int, default=None,
                       help="Choose 'N' random samples")
    
    parser.add_argument("--seed", type=int, default=None,
                       help="Seed for random shuffling")

    parser.add_argument("--save_path", type=str, required=True,
                       help="Path to save the final trees")

    parser.add_argument("--max_seq_length", default=200, type=int,
                    help="The maximum total input sequence length after tokenization")

    args = parser.parse_args()
    
    inside_model = InsideOutsideStringPredictor(model_name_or_path=args.model_name_or_path, 
                                                pre_trained_model_path=args.pre_trained_model_path,
                                                max_length=args.max_seq_length)
    
    with open(args.save_path, "w") as out_file:
        
        if args.predict_on_train:
            train_sentences = DataLoaderHelper(input_file_object=PTB_TRAIN_SENTENCES_WITHOUT_PUNCTUATION_PATH).read_lines()
            train_gold_file_path = PTB_TRAIN_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH
            
        if args.predict_on_test:
            test_sentences = DataLoaderHelper(input_file_object=PTB_TEST_SENTENCES_WITHOUT_PUNCTUATION_PATH).read_lines()
            test_gold_file_path = PTB_TEST_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH

        if args.randomN:
            train_sentences = train_sentences[:args.randomN]
            random.seed(args.seed)
            random.shuffle(train_sentences)
            
        if args.predict_on_test:
            for test_index, test_sentence in enumerate(test_sentences):
                best_parse = process_test_sample(test_index, test_sentence, test_gold_file_path, inside_model=inside_model)
                out_file.write(best_parse + "\n")
                
        if args.predict_on_train:
            for train_index, train_sentence in enumerate(train_sentences):
                bootstrap_data = process_train_sample(train_sentence, seed=args.seed)
                if train_index == 0:
                    bootstrap_data.to_csv(INSIDE_BOOTSTRAPPED_DATASET_PATH + "self_train_1.csv", index=False, mode="a")
                else:
                    bootstrap_data.to_csv(INSIDE_BOOTSTRAPPED_DATASET_PATH + "self_train_1.csv", index=False, header=None, mode="a")
        
if __name__ == "__main__":
    main()
    
# python weakly_supervised_parser/inference.py --predict_on_test --model_name_or_path roberta-base --pre_trained_model_path weakly_supervised_parser/model/BOOTSTRAP_DATA/INSIDE/inside.onnx  --max_seq_length 200  --save_path TEMP/predictions/english/inside_model_predictions.txt
# python weakly_supervised_parser/inference.py --predict_on_train --save_path TEMP/predictions/english/inside_train_predictions.txt
