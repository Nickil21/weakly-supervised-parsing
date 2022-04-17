from argparse import ArgumentParser
from loguru import logger

from weakly_supervised_parser.settings import TRAINED_MODEL_PATH
from weakly_supervised_parser.utils.prepare_dataset import DataLoaderHelper
from weakly_supervised_parser.utils.populate_chart import PopulateCKYChart
from weakly_supervised_parser.tree.evaluate import calculate_F1_for_spans, tree_to_spans
from weakly_supervised_parser.model.trainer import InsideOutsideStringClassifier
from weakly_supervised_parser.settings import PTB_TEST_SENTENCES_WITHOUT_PUNCTUATION_PATH, PTB_TEST_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH


class Predictor:
    def __init__(self, sentence):
        self.sentence = sentence
        self.sentence_list = sentence.split()

    def obtain_best_parse(self, predict_type, model, scale_axis, predict_batch_size, return_df=False):
        unique_tokens_flag, span_scores, df = PopulateCKYChart(sentence=self.sentence).fill_chart(predict_type=predict_type, 
                                                                                                  model=model, 
                                                                                                  scale_axis=scale_axis, 
                                                                                                  predict_batch_size=predict_batch_size)

        if unique_tokens_flag:
            best_parse = "(S " + " ".join(["(S " + item + ")" for item in self.sentence_list]) + ")"
            logger.info("BEST PARSE", best_parse)
        else:
            best_parse = PopulateCKYChart(sentence=self.sentence).best_parse_tree(span_scores)
        if return_df:
            return best_parse, df
        return best_parse


def process_test_sample(index, sentence, gold_file_path, predict_type, model, scale_axis, predict_batch_size, return_df=False):
    best_parse, df = Predictor(sentence=sentence).obtain_best_parse(predict_type=predict_type, 
                                                                    model=model, 
                                                                    scale_axis=scale_axis, 
                                                                    predict_batch_size=predict_batch_size,
                                                                    return_df=True)
    gold_standard = DataLoaderHelper(input_file_object=gold_file_path)
    sentence_f1 = calculate_F1_for_spans(tree_to_spans(gold_standard[index]), tree_to_spans(best_parse))
    if sentence_f1 < 25.0:
        logger.warning(f"Index: {index} <> F1: {sentence_f1:.2f}")
    else:
        logger.info(f"Index: {index} <> F1: {sentence_f1:.2f}")
    if return_df:
        return best_parse, df
    else:
        return best_parse


def process_co_train_test_sample(index, sentence, gold_file_path, inside_model, outside_model, return_df=False):
    _, df_inside = PopulateCKYChart(sentence=sentence).compute_scores(predict_type="inside", model=inside_model, return_df=True)
    _, df_outside = PopulateCKYChart(sentence=sentence).compute_scores(predict_type="outside", model=outside_model, return_df=True)
    df = df_inside.copy()
    df["scores"] = df_inside["scores"] * df_outside["scores"]
    _, span_scores, df = PopulateCKYChart(sentence=sentence).fill_chart(data=df)
    best_parse = PopulateCKYChart(sentence=sentence).best_parse_tree(span_scores)
    gold_standard = DataLoaderHelper(input_file_object=gold_file_path)
    sentence_f1 = calculate_F1_for_spans(tree_to_spans(gold_standard[index]), tree_to_spans(best_parse))
    if sentence_f1 < 25.0:
        logger.warning(f"Index: {index} <> F1: {sentence_f1:.2f}")
    else:
        logger.info(f"Index: {index} <> F1: {sentence_f1:.2f}")
    return best_parse


def main():
    parser = ArgumentParser(description="Inference Pipeline for the Inside Outside String Classifier", add_help=True)

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument("--use_inside", action="store_true", help="Whether to predict using inside model")

    group.add_argument("--use_inside_self_train", action="store_true", help="Whether to predict using inside model with self-training")

    group.add_argument("--use_outside", action="store_true", help="Whether to predict using outside model")

    group.add_argument("--use_inside_outside_co_train", action="store_true", help="Whether to predict using inside-outside model with co-training")

    parser.add_argument("--model_name_or_path", type=str, default="roberta-base", help="Path to the model identifier from huggingface.co/models")

    parser.add_argument("--save_path", type=str, required=True, help="Path to save the final trees")
    
    parser.add_argument("--scale_axis", choices=[None, 1], default=None, help="Whether to scale axis globally (None) or sequentially (1) across batches during softmax computation")
    
    parser.add_argument("--predict_batch_size", type=int, help="Batch size during inference")

    parser.add_argument(
        "--inside_max_seq_length", default=256, type=int, help="The maximum total input sequence length after tokenization for the inside model"
    )

    parser.add_argument(
        "--outside_max_seq_length", default=64, type=int, help="The maximum total input sequence length after tokenization for the outside model"
    )

    args = parser.parse_args()

    if args.use_inside:
        pre_trained_model_path = TRAINED_MODEL_PATH + "inside_model.onnx"
        max_seq_length = args.inside_max_seq_length

    if args.use_inside_self_train:
        pre_trained_model_path = TRAINED_MODEL_PATH + "inside_model_self_trained.onnx"
        max_seq_length = args.inside_max_seq_length

    if args.use_outside:
        pre_trained_model_path = TRAINED_MODEL_PATH + "outside_model.onnx"
        max_seq_length = args.outside_max_seq_length

    if args.use_inside_outside_co_train:
        inside_pre_trained_model_path = "inside_model_co_trained.onnx"
        inside_model = InsideOutsideStringClassifier(model_name_or_path=args.model_name_or_path, max_seq_length=args.inside_max_seq_length)
        inside_model.load_model(pre_trained_model_path=inside_pre_trained_model_path)

        outside_pre_trained_model_path = "outside_model_co_trained.onnx"
        outside_model = InsideOutsideStringClassifier(model_name_or_path=args.model_name_or_path, max_seq_length=args.outside_max_seq_length)
        outside_model.load_model(pre_trained_model_path=outside_pre_trained_model_path)
    else:
        model = InsideOutsideStringClassifier(model_name_or_path=args.model_name_or_path, max_seq_length=max_seq_length)
        model.load_model(pre_trained_model_path=pre_trained_model_path)

    if args.use_inside or args.use_inside_self_train:
        predict_type = "inside"

    if args.use_outside:
        predict_type = "outside"

    with open(args.save_path, "w") as out_file:
        print(type(args.scale_axis))
        test_sentences = DataLoaderHelper(input_file_object=PTB_TEST_SENTENCES_WITHOUT_PUNCTUATION_PATH).read_lines()
        test_gold_file_path = PTB_TEST_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH
        for test_index, test_sentence in enumerate(test_sentences):
            if args.use_inside_outside_co_train:
                best_parse = process_co_train_test_sample(
                    test_index, test_sentence, test_gold_file_path, inside_model=inside_model, outside_model=outside_model
                )
            else:
                best_parse = process_test_sample(test_index, test_sentence, test_gold_file_path, predict_type=predict_type, model=model,
                                                 scale_axis=args.scale_axis, predict_batch_size=args.predict_batch_size)

            out_file.write(best_parse + "\n")


if __name__ == "__main__":
    main()
