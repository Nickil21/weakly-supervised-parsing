import re
import numpy as np
from weakly_supervised_parser.tree.helpers import Tree


def CKY(sent_all, prob_s, label_s, verbose=False):
    r"""
    choose tree with maximum expected number of constituents,
    or max \sum_{(i,j) \in tree} p((i,j) is constituent)
    """

    def backpt_to_tree(sent, backpt, label_table):
        def to_tree(i, j):
            if j - i == 1:
                return Tree(sent[i], None, sent[i])
            else:
                k = backpt[i][j]
                return Tree(label_table[i][j], [to_tree(i, k), to_tree(k, j)], None)

        return to_tree(0, len(sent))

    def to_table(value_s, i_s, j_s):
        table = [[None for _ in range(np.max(j_s) + 1)] for _ in range(np.max(i_s) + 1)]
        for value, i, j in zip(value_s, i_s, j_s):
            table[i][j] = value
        return table

    # produce list of spans to pass to is_constituent, while keeping track of which sentence
    sent_s, i_s, j_s = [], [], []
    idx_all = []
    for sent in sent_all:
        start = len(sent_s)
        for i in range(len(sent)):
            for j in range(i + 1, len(sent) + 1):
                sent_s.append(sent)
                i_s.append(i)
                j_s.append(j)
        idx_all.append((start, len(sent_s)))

    # feed spans to is_constituent
    # prob_s, label_s = self.is_constituent(sent_s, i_s, j_s, verbose = verbose)

    # given span probs, perform CKY to get best tree for each sentence.
    tree_all, prob_all = [], []
    for sent, idx in zip(sent_all, idx_all):
        # first, use tables to keep track of things
        k, l = idx
        prob, label = prob_s[k:l], label_s[k:l]
        i, j = i_s[k:l], j_s[k:l]

        prob_table = to_table(prob, i, j)
        label_table = to_table(label, i, j)

        # perform cky using scores and backpointers
        score_table = [[None for _ in range(len(sent) + 1)] for _ in range(len(sent))]
        backpt_table = [[None for _ in range(len(sent) + 1)] for _ in range(len(sent))]
        for i in range(len(sent)):  # base case: single words
            score_table[i][i + 1] = 1
        for j in range(2, len(sent) + 1):
            for i in range(j - 2, -1, -1):
                best, argmax = -np.inf, None
                for k in range(i + 1, j):  # find splitpoint
                    score = score_table[i][k] + score_table[k][j]
                    if score > best:
                        best, argmax = score, k
                score_table[i][j] = best + prob_table[i][j]
                backpt_table[i][j] = argmax

        tree = backpt_to_tree(sent, backpt_table, label_table)
        tree_all.append(tree)
        prob_all.append(prob_table)

    return tree_all, prob_all


def get_best_parse(sentence, spans):
    flattened_scores = []
    for i in range(spans.shape[0]):
        for j in range(spans.shape[1]):
            if i > j:
                continue
            else:
                flattened_scores.append(spans[i, j])
    prob_s, label_s = flattened_scores, ["S"] * len(flattened_scores)
    # print(prob_s, label_s)
    trees, _ = CKY(sent_all=sentence, prob_s=prob_s, label_s=label_s)
    s = str(trees[0])
    # Replace previous occurrence of string
    out = re.sub(r"(?<![^\s()])([^\s()]+)(?=\s+\1(?![^\s()]))", "S", s)
    # best_parse = "(ROOT " + out + ")"
    return out  # best_parse
