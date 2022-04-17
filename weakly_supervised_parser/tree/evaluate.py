import argparse
import collections
import os
import subprocess

import nltk


def tree_to_spans(tree, keep_labels=False, keep_leaves=False, keep_whole_span=False):
    if isinstance(tree, str):
        tree = nltk.Tree.fromstring(tree)

    length = len(tree.pos())
    queue = collections.deque(tree.treepositions())
    stack = [(queue.popleft(), 0)]
    j = 0
    spans = []
    while stack != []:
        (p, i) = stack[-1]
        if not queue or queue[0][:-1] != p:
            if isinstance(tree[p], nltk.tree.Tree):
                if j - i > 1:
                    spans.append((tree[p].label(), (i, j)))
            else:
                j = i + 1
            stack.pop()
        else:
            q = queue.popleft()
            stack.append((q, j))
    if not keep_whole_span:
        spans = [span for span in spans if span[1] != (0, length)]
    if not keep_labels:
        spans = [span[1] for span in spans]
    return spans


def test_tree_to_spans():
    assert [(0, 2), (0, 3), (0, 4)] == tree_to_spans("(S (S (S (S (S 1) (S 2)) (S 3)) (S 4)) (S 5))", keep_labels=False)
    assert [] == tree_to_spans("(S 1)", keep_labels=False)
    assert [] == tree_to_spans("(S (S 1) (S 2))", keep_labels=False)
    assert [(1, 3)] == tree_to_spans("(S (S 1) (S (S 2) (S 3)))", keep_labels=False)
    assert [("S", (1, 3))] == tree_to_spans("(S (S 1) (S (S 2) (S 3)))", keep_labels=True)


def get_F1_score_intermediates(gold_spans, pred_spans):
    """Get intermediate results for calculating the F1 score"""
    n_true_positives = 0
    gold_span_counter = collections.Counter(gold_spans)
    pred_span_counter = collections.Counter(pred_spans)
    unique_spans = set(gold_spans + pred_spans)
    for span in unique_spans:
        n_true_positives += min(gold_span_counter[span], pred_span_counter[span])
    return n_true_positives, len(gold_spans), len(pred_spans)


def calculate_F1_score_from_intermediates(n_true_positives, n_golds, n_predictions, precision_recall_f_score=False):
    """Calculate F1 score"""
    if precision_recall_f_score:
        zeros = (0, 0, 0)
    else:
        zeros = 0
    if n_golds == 0:
        return 100 if n_predictions == 0 else zeros
    if n_true_positives == 0 or n_predictions == 0:
        return zeros
    recall = n_true_positives / n_golds
    precision = n_true_positives / n_predictions
    F1 = 2 * precision * recall / (precision + recall)
    if precision_recall_f_score:
        return precision, recall, F1 * 100
    return F1 * 100


def calculate_F1_for_spans(gold_spans, pred_spans, precision_recall_f_score=False):
    #  CHANGE THIS LATER
    # gold_spans = list(set(gold_spans))
    ###################################
    tp, n_gold, n_pred = get_F1_score_intermediates(gold_spans, pred_spans)
    if precision_recall_f_score:
        p, r, F1 = calculate_F1_score_from_intermediates(tp, len(gold_spans), len(pred_spans), precision_recall_f_score=precision_recall_f_score)
        return p, r, F1
    F1 = calculate_F1_score_from_intermediates(tp, len(gold_spans), len(pred_spans))
    return F1


def test_calculate_F1_for_spans():
    pred = [(0, 1)]
    gold = [(0, 1)]
    assert calculate_F1_for_spans(gold, pred) == 100
    pred = [(0, 0)]
    gold = [(0, 1)]
    assert calculate_F1_for_spans(gold, pred) == 0
    pred = [(0, 0), (0, 1)]
    gold = [(0, 1), (1, 1)]
    assert calculate_F1_for_spans(gold, pred) == 50
    pred = [(0, 0), (0, 0)]
    gold = [(0, 0), (0, 0), (0, 1)]
    assert calculate_F1_for_spans(gold, pred) == 80
    pred = [(0, 0), (1, 0)]
    gold = [(0, 0), (0, 0), (0, 1)]
    assert calculate_F1_for_spans(gold, pred) == 40


def read_lines_from_file(filepath, len_limit):
    with open(filepath, "r") as f:
        for line in f:
            tree = nltk.Tree.fromstring(line)
            if len_limit is not None and len(tree.pos()) > len_limit:
                continue
            yield line.strip()


def read_spans_from_file(filepath, len_limit):
    for line in read_lines_from_file(filepath, len_limit):
        yield tree_to_spans(line, keep_labels=False, keep_leaves=False, keep_whole_span=False)


def calculate_corpus_level_F1_for_spans(gold_list, pred_list):
    n_true_positives = 0
    n_golds = 0
    n_predictions = 0
    for gold_spans, pred_spans in zip(gold_list, pred_list):
        n_tp, n_g, n_p = get_F1_score_intermediates(gold_spans, pred_spans)
        n_true_positives += n_tp
        n_golds += n_g
        n_predictions += n_p
    F1 = calculate_F1_score_from_intermediates(n_true_positives, n_golds, n_predictions)
    return F1


def calculate_sentence_level_F1_for_spans(gold_list, pred_list):
    f1_scores = []
    for gold_spans, pred_spans in zip(gold_list, pred_list):
        f1 = calculate_F1_for_spans(gold_spans, pred_spans)
        f1_scores.append(f1)
    F1 = sum(f1_scores) / len(f1_scores)
    return F1


def parse_evalb_results_from_file(filepath):
    i_th_score = 0
    score_of_all_length = None
    score_of_length_10 = None
    prefix_of_the_score_line = "Bracketing FMeasure       ="

    with open(filepath, "r") as f:
        for line in f:
            if line.startswith(prefix_of_the_score_line):
                i_th_score += 1
                if i_th_score == 1:
                    score_of_all_length = float(line.split()[-1])
                elif i_th_score == 2:
                    score_of_length_10 = float(line.split()[-1])
                else:
                    raise ValueError("Too many lines for F score")
    return score_of_all_length, score_of_length_10


def execute_evalb(gold_file, pred_file, out_file, len_limit):
    EVALB_PATH = "model/EVALB/"
    subprocess.run("{} -p {} {} {} > {}".format(EVALB_PATH + "/evalb", EVALB_PATH + "unlabelled.prm", gold_file, pred_file, out_file), shell=True)


def calculate_evalb_F1_for_file(gold_file, pred_file, len_limit):
    evalb_out_file = pred_file + ".evalb_out"
    execute_evalb(gold_file, pred_file, evalb_out_file, len_limit)
    F1_len_all, F1_len_10 = parse_evalb_results_from_file(evalb_out_file)
    if len_limit is None:
        return F1_len_all
    elif len_limit == 10:
        return F1_len_10
    else:
        raise ValueError(f"Unexpected len_limit: {len_limit}")


def calculate_sentence_level_F1_for_file(gold_file, pred_file, len_limit):
    gold_list = list(read_spans_from_file(gold_file, len_limit))
    pred_list = list(read_spans_from_file(pred_file, len_limit))
    F1 = calculate_sentence_level_F1_for_spans(gold_list, pred_list)
    return F1


def calculate_corpus_level_F1_for_file(gold_file, pred_file, len_limit):
    gold_list = list(read_spans_from_file(gold_file, len_limit))
    pred_list = list(read_spans_from_file(pred_file, len_limit))
    F1 = calculate_corpus_level_F1_for_spans(gold_list, pred_list)
    return F1


def evaluate_prediction_file(gold_file, pred_file, len_limit):
    corpus_F1 = calculate_corpus_level_F1_for_file(gold_file, pred_file, len_limit)
    sentence_F1 = calculate_sentence_level_F1_for_file(gold_file, pred_file, len_limit)
    # evalb_F1 = calculate_evalb_F1_for_file(gold_file, pred_file, len_limit)

    print("=====> Evaluation Results <=====")
    print(f"Length constraint: f{len_limit}")
    print(f"Micro F1: {corpus_F1:.2f}, Macro F1: {sentence_F1:.2f}")  # , evalb_F1))
    print("=====> Evaluation Results <=====")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_file", "-g", help="path to gold file")
    parser.add_argument("--pred_file", "-p", help="path to prediction file")
    parser.add_argument(
        "--len_limit", default=None, type=int, choices=(None, 10, 20, 30, 40, 50, 100), help="length constraint for evaluation, 10 or None"
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    evaluate_prediction_file(args.gold_file, args.pred_file, args.len_limit)


if __name__ == "__main__":
    main()

#  python helper/evaluate.py -g TEMP/preprocessed_dev.txt -p TEMP/pred_dev_m_None.txt
