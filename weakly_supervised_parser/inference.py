from argparse import ArgumentParser

from weakly_supervised_parser.utils.prepare_dataset import DataLoaderHelper
from weakly_supervised_parser.utils.cky_algorithm import get_best_parse
from weakly_supervised_parser.utils.populate_chart import PopulateCKYChart
from weakly_supervised_parser.tree.evaluate import calculate_F1_for_spans, tree_to_spans
from weakly_supervised_parser.tree.helpers import get_constituents
from weakly_supervised_parser.settings import PTB_TEST_SENTENCES_WITHOUT_PUNCTUATION_PATH, PTB_TEST_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH
from weakly_supervised_parser.settings import PTB_TRAIN_SENTENCES_WITHOUT_PUNCTUATION_PATH, PTB_TRAIN_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH


class Predictor:
    def __init__(self, sentence):
        self.sentence = sentence
        self.sentence_list = sentence.split()

    def obtain_best_parse(self):
        flags, span_scores = PopulateCKYChart(sentence=self.sentence).fill_chart()
        if flags:
            best_parse = "(S " + " ".join(["(S " + item + ")" for item in self.sentence_list]) + ")"
        else:
            span_scores_cky_format = span_scores[:-1, 1:]
            best_parse = get_best_parse(sentence=[self.sentence_list], spans=span_scores_cky_format)
        return best_parse


def process_sample(index, sentence, gold_file_path):
    best_parse = Predictor(sentence=sentence).obtain_best_parse()
    gold_standard = DataLoaderHelper(input_file_object=gold_file_path)
    sentence_f1 = calculate_F1_for_spans(tree_to_spans(gold_standard[index]), tree_to_spans(best_parse))
    return best_parse


def main():
    parser = ArgumentParser(description="Inference Pipeline for the Inside Outside String Classifier", add_help=True)

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument("--predict_on_train", action="store_true", help="Whether to predict on Train or not")

    group.add_argument("--predict_on_test", action="store_true", help="Whether to predict on Test or not")

    parser.add_argument("--topN", type=int, default=None, help="Top (N) number of predictions")

    parser.add_argument("--save_path", type=str, required=True, help="Path to save the final trees")

    args = parser.parse_args()

    with open(args.save_path, "w") as out_file:
        if args.predict_on_train:
            sentences = DataLoaderHelper(input_file_object=PTB_TRAIN_SENTENCES_WITHOUT_PUNCTUATION_PATH).read_lines()
            gold_file_path = PTB_TRAIN_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH
        if args.predict_on_test:
            sentences = DataLoaderHelper(input_file_object=PTB_TEST_SENTENCES_WITHOUT_PUNCTUATION_PATH).read_lines()
            gold_file_path = PTB_TEST_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH
        if args.topN:
            sentences = sentences[: args.topN]
        for index, sentence in enumerate(sentences):
            best_parse = process_sample(index, sentence, gold_file_path)
            out_file.write(best_parse + "\n")


if __name__ == "__main__":
    main()

#  python weakly_supervised_parser/inference.py --predict_on_test --topN 100 --save_path TEMP/predictions/english/inside_model_predictions.txt
