import nltk
import pandas as pd
from collections import Counter
from weakly_supervised_parser.tree.evaluate import tree_to_spans


class Tree(object):
    def __init__(self, label, children, word):
        self.label = label
        self.children = children
        self.word = word

    def __str__(self):
        return self.linearize()

    def linearize(self):
        if not self.children:
            return f"({self.label} {self.word})"
        return f"({self.label} {' '.join(c.linearize() for c in self.children)})"

    def spans(self, start=0):
        if not self.children:
            return [(start, start + 1)]
        span_list = []
        position = start
        for c in self.children:
            cspans = c.spans(start=position)
            span_list.extend(cspans)
            position = cspans[0][1]
        return [(start, position)] + span_list

    def spans_labels(self, start=0):
        if not self.children:
            return [(start, start + 1, self.label)]
        span_list = []
        position = start
        for c in self.children:
            cspans = c.spans_labels(start=position)
            span_list.extend(cspans)
            position = cspans[0][1]
        return [(start, position, self.label)] + span_list


def extract_sentence(sentence):
    t = nltk.Tree.fromstring(sentence)
    return " ".join(item[0] for item in t.pos())


def get_constituents(sample_string, want_spans_mapping=False, whole_sentence=True, labels=False):
    t = nltk.Tree.fromstring(sample_string)
    if want_spans_mapping:
        spans = tree_to_spans(t, keep_labels=True)
        return dict(Counter(item[1] for item in spans))
    spans = tree_to_spans(t, keep_labels=True)
    sentence = extract_sentence(sample_string).split()

    labeled_consituents_lst = []
    constituents = []
    for span in spans:
        labeled_consituents = {}
        labeled_consituents["labels"] = span[0]
        i, j = span[1][0], span[1][1]
        constituents.append(" ".join(sentence[i:j]))
        labeled_consituents["constituent"] = " ".join(sentence[i:j])
        labeled_consituents_lst.append(labeled_consituents)

    # Add original sentence
    if whole_sentence:
        constituents = constituents + [" ".join(sentence)]

    if labels:
        return labeled_consituents_lst

    return constituents


def get_distituents(sample_string):
    sentence = extract_sentence(sample_string).split()

    def get_all_combinations(sentence):
        L = sentence.split()
        N = len(L)
        out = []
        for n in range(2, N):
            for i in range(N - n + 1):
                out.append((i, i + n))
        return out

    combinations = get_all_combinations(extract_sentence(sample_string))
    constituents = list(get_constituents(sample_string, want_spans_mapping=True).keys())
    spans = [item for item in combinations if item not in constituents]
    distituents = []
    for span in spans:
        i, j = span[0], span[1]
        distituents.append(" ".join(sentence[i:j]))
    return distituents


def get_leaves(tree):
    if not tree.children:
        return [tree]
    leaves = []
    for c in tree.children:
        leaves.extend(get_leaves(c))
    return leaves


def unlinearize(string):
    """
    (TOP (S (NP (PRP He)) (VP (VBD was) (ADJP (JJ right))) (. .)))
    """
    tokens = string.replace("(", " ( ").replace(")", " ) ").split()

    def read_tree(start):
        if tokens[start + 2] != "(":
            return Tree(tokens[start + 1], None, tokens[start + 2]), start + 4
        i = start + 2
        children = []
        while tokens[i] != ")":
            tree, i = read_tree(i)
            children.append(tree)
        return Tree(tokens[start + 1], children, None), i + 1

    tree, _ = read_tree(0)
    return tree


def recall_by_label(gold_standard, best_parse):
    correct = {}
    total = {}
    for tree1, tree2 in zip(gold_standard, best_parse):
        try:
            leaves1, leaves2 = get_leaves(tree1["tree"]), get_leaves(tree2["tree"])
            for l1, l2 in zip(leaves1, leaves2):
                assert l1.word.lower() == l2.word.lower(), f"{l1.word} =/= {l2.word}"
            spanlabels = tree1["tree"].spans_labels()
            spans = tree2["tree"].spans()

            for (i, j, label) in spanlabels:
                if j - i != 1:
                    if label not in correct:
                        correct[label] = 0
                        total[label] = 0
                    if (i, j) in spans:
                        correct[label] += 1
                    total[label] += 1
        except Exception as e:
            print(e)
    acc = {}
    for label in total.keys():
        acc[label] = correct[label] / total[label]
    return acc


def label_recall_output(gold_standard, best_parse):
    best_parse_trees = []
    gold_standard_trees = []
    for t1, t2 in zip(gold_standard, best_parse):
        gold_standard_trees.append({"tree": unlinearize(t1)})
        best_parse_trees.append({"tree": unlinearize(t2)})

    dct = recall_by_label(gold_standard=gold_standard_trees, best_parse=best_parse_trees)
    labels = ["SBAR", "NP", "VP", "PP", "ADJP", "ADVP"]
    l = [{label: f"{recall * 100:.2f}"} for label, recall in dct.items() if label in labels]
    df = pd.DataFrame([item.values() for item in l], index=[item.keys() for item in l], columns=["recall"])
    df.index = df.index.map(lambda x: list(x)[0])
    df_out = df.reindex(labels)
    return df_out


if __name__ == "__main__":
    # import pandas as pd
    # from weakly_supervised_parser.utils.prepare_dataset import PTBDataset
    # from weakly_supervised_parser.settings import PTB_TEST_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH, PTB_SAVE_TREES_PATH

    # best_parse = PTBDataset(PTB_SAVE_TREES_PATH + "inside_model_predictions.txt").retrieve_all_sentences()
    # gold_standard = PTBDataset(PTB_TEST_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH).retrieve_all_sentences()
    # print(label_recall_output(gold_standard, best_parse))
    s = "(IP (CONJ しかし) (PP (NP (PP (NP (NUM 二) (CL 度目)) (P の)) (PP (NP 車輪) (P の)) (N 音)) (P は)) (PU 、) (ADVP もう) (PP (NP 彼) (P を)) (VB 驚かさ) (NEG なかっ) (AXD た) (PU 。))"
    print(get_constituents(s)[-1])
