from collections import defaultdict, Counter
from weakly_supervised_parser.tree import evaluate, helpers



def to_tree(phrase, spans):
    words = phrase.split()
    iter = reversed(spans)
    current = None

    def encode(start, end):
        return ["(S {})".format(words[k]) for k in range(end - 1, start - 1, -1)]

    def recur(start, end, tab=""):
        nonlocal current
        nodes = []
        current = next(iter, None)
        while current and current[1] > start:  # overlap => it's a child
            child_start, child_end = current
            assert child_start >= start and child_end <= end, "Invalid spans"
            # encode what comes at the right of this child (single words):
            nodes.extend(encode(child_end, end))
            # encode the child itself using recursion
            nodes.append(recur(child_start, child_end, tab+"  "))
            end = child_start
        nodes.extend(encode(start, end))
        return "(S {})".format(" ".join(reversed(nodes))) if len(nodes) > 1 else nodes[0]

    return "{}".format(recur(0, len(words)))



class RuleBasedHeuristic:

    def __init__(self, sentence=None, corpus=None):
        self.sentence = sentence
        self.corpus = corpus

    def add_contiguous_titlecase_words(self, row):
        matches = []
        dd = defaultdict(list)
        count = 0
        for i, j in zip(row, row[1:]):
            if j[0] - i[0] == 1:
                dd[count].append(i[-1] + " " + j[-1])
            else:
                count += 1
        for key, value in dd.items():
            if len(value) > 1:
                out = value[0]
                inter = ""
                for item in value[1:]:
                    inter += " " + item.split()[-1]
                matches.append(out + inter)
            else:
                matches.extend(value)
        return matches

    def augment_using_most_frequent_starting_token(self):
        first_token = []
        for sentence in self.corpus:
            first_token.append(sentence.split()[0])
        return Counter(first_token).most_common(1)[0][0]
    
    def delete_false_constituents(self, tree, start_exclude_list=None, end_exclude_list=None):
        start_exclude_list = [item.lower() for item in start_exclude_list]
        end_exclude_list = [item.lower() for item in end_exclude_list]
        best_parse_constituents = helpers.get_constituents(tree)
        best_parse_constituents_spans = evaluate.tree_to_spans(tree)
        
        delete_best_parse_constituents_inc_exc = [constituent for constituent in best_parse_constituents if constituent.startswith(tuple(start_exclude_list)) or constituent.endswith(tuple(end_exclude_list))]
        delete_best_parse_constituents = delete_best_parse_constituents_inc_exc #pp(best_parse_constituents) + delete_best_parse_constituents_inc_exc #+ list(delete)
        sentence_length = len(self.sentence.split())
        delete_constituent_spans = []
        for delete_constituent in delete_best_parse_constituents:
            for n in range(2,  sentence_length + 1):
                for i in range(sentence_length - n + 1):
                    if " ".join(self.sentence.split()[i: i + n]) == delete_constituent:
                        # print((i, i + n), delete_constituent)
                        delete_constituent_spans.append((i, i+n))

        spans = [span for span in best_parse_constituents_spans if span not in delete_constituent_spans]
        out = to_tree(self.sentence, spans)
        # print(out)
        return out



if __name__ == "__main__":
    from weakly_supervised_parser.utils.prepare_dataset import PTBDataset
    from weakly_supervised_parser.settings import PTB_TRAIN_SENTENCES_WITH_PUNCTUATION_PATH

    ptb = PTBDataset(training_data_path=PTB_TRAIN_SENTENCES_WITH_PUNCTUATION_PATH)
    print(RuleBasedHeuristic().augment_using_most_frequent_starting_token(sentences=ptb.retrieve_all_sentences()))