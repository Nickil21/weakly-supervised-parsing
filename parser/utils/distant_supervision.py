from collections import defaultdict, Counter


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

if __name__ == "__main__":
    from parser.utils.prepare_dataset import PTBDataset
    from parser.settings import PTB_TRAIN_SENTENCES_WITH_PUNCTUATION_PATH

    ptb = PTBDataset(training_data_path=PTB_TRAIN_SENTENCES_WITH_PUNCTUATION_PATH)
    print(RuleBasedHeuristic().augment_using_most_frequent_starting_token(sentences=ptb.retrieve_all_sentences()))