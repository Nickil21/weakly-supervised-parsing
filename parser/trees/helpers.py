import nltk
from collections import Counter
from parser.trees.evaluate import tree_to_spans


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
        labeled_consituents['labels'] = span[0]
        i, j = span[1][0], span[1][1]
        constituents.append(" ".join(sentence[i:j]))
        labeled_consituents['constituent'] = " ".join(sentence[i:j])
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