import argparse
import os
import copy
import re
import sys

import pandas as pd

from nltk.corpus import ptb

from weakly_supervised_parser.settings import (
    PTB_TRAIN_GOLD_WITHOUT_PUNCTUATION_PATH,
    PTB_VALID_GOLD_WITHOUT_PUNCTUATION_PATH,
    PTB_TEST_GOLD_WITHOUT_PUNCTUATION_PATH,
)
from weakly_supervised_parser.settings import (
    PTB_TRAIN_SENTENCES_WITH_PUNCTUATION_PATH,
    PTB_VALID_SENTENCES_WITH_PUNCTUATION_PATH,
    PTB_TEST_SENTENCES_WITH_PUNCTUATION_PATH,
)
from weakly_supervised_parser.settings import (
    PTB_TRAIN_SENTENCES_WITHOUT_PUNCTUATION_PATH,
    PTB_VALID_SENTENCES_WITHOUT_PUNCTUATION_PATH,
    PTB_TEST_SENTENCES_WITHOUT_PUNCTUATION_PATH,
)
from weakly_supervised_parser.settings import (
    PTB_TRAIN_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH,
    PTB_VALID_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH,
    PTB_TEST_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH,
)
from weakly_supervised_parser.settings import (
    YOON_KIM_TRAIN_GOLD_WITHOUT_PUNCTUATION_PATH,
    YOON_KIM_VALID_GOLD_WITHOUT_PUNCTUATION_PATH,
    YOON_KIM_TEST_GOLD_WITHOUT_PUNCTUATION_PATH,
)

from weakly_supervised_parser.tree.helpers import extract_sentence


class AlignPTBYoonKimFormat:
    def __init__(self, ptb_data_path, yk_data_path):
        self.ptb_data = pd.read_csv(ptb_data_path, sep="\t", header=None)
        self.yk_data = pd.read_csv(yk_data_path, sep="\t", header=None)

    def row_mapper(self, save_data_path):
        dict_mapper = self.ptb_data.reset_index().merge(self.yk_data.reset_index(), on=[0]).set_index("index_y")["index_x"].to_dict()
        self.ptb_data.loc[self.ptb_data.index.map(dict_mapper)].to_csv(save_data_path, sep="\t", index=False, header=None)
        return dict_mapper


currency_tags_words = ["#", "$", "C$", "A$"]
ellipsis = ["*", "*?*", "0", "*T*", "*ICH*", "*U*", "*RNR*", "*EXP*", "*PPA*", "*NOT*"]
punctuation_tags = [".", ",", ":", "-LRB-", "-RRB-", "''", "``"]
punctuation_words = [".", ",", ":", "-LRB-", "-RRB-", "''", "``", "--", ";", "-", "?", "!", "...", "-LCB-", "-RCB-"]


def get_data_ptb(root, output):
    # tag filter is from https://github.com/yikangshen/PRPN/blob/master/data_ptb.py
    word_tags = [
        "CC",
        "CD",
        "DT",
        "EX",
        "FW",
        "IN",
        "JJ",
        "JJR",
        "JJS",
        "LS",
        "MD",
        "NN",
        "NNS",
        "NNP",
        "NNPS",
        "PDT",
        "POS",
        "PRP",
        "PRP$",
        "RB",
        "RBR",
        "RBS",
        "RP",
        "SYM",
        "TO",
        "UH",
        "VB",
        "VBD",
        "VBG",
        "VBN",
        "VBP",
        "VBZ",
        "WDT",
        "WP",
        "WP$",
        "WRB",
    ]
    train_file_ids = []
    val_file_ids = []
    test_file_ids = []
    train_section = ["02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]
    val_section = ["22"]
    test_section = ["23"]

    for dir_name, _, file_list in os.walk(root, topdown=False):
        if dir_name.split("/")[-1] in train_section:
            file_ids = train_file_ids
        elif dir_name.split("/")[-1] in val_section:
            file_ids = val_file_ids
        elif dir_name.split("/")[-1] in test_section:
            file_ids = test_file_ids
        else:
            continue
        for fname in file_list:
            file_ids.append(os.path.join(dir_name, fname))
            assert file_ids[-1].split(".")[-1] == "mrg"
    print(len(train_file_ids), len(val_file_ids), len(test_file_ids))

    def del_tags(tree, word_tags):
        for sub in tree.subtrees():
            for n, child in enumerate(sub):
                if isinstance(child, str):
                    continue
                if all(leaf_tag not in word_tags for leaf, leaf_tag in child.pos()):
                    del sub[n]

    def save_file(file_ids, out_file, include_punctuation=False):
        f_out = open(out_file, "w")
        for f in file_ids:
            sentences = ptb.parsed_sents(f)
            for sen_tree in sentences:
                sen_tree_copy = copy.deepcopy(sen_tree)
                c = 0
                while not all([tag in word_tags for _, tag in sen_tree.pos()]):
                    del_tags(sen_tree, word_tags)
                    c += 1
                    if c > 10:
                        assert False

                if len(sen_tree.leaves()) < 2:
                    print(f"skipping {' '.join(sen_tree.leaves())} since length < 2")
                    continue

                if include_punctuation:
                    keep_punctuation_tags = word_tags + punctuation_tags
                    out = " ".join([token for token, pos_tag in sen_tree_copy.pos() if pos_tag in keep_punctuation_tags])
                else:
                    out = sen_tree.pformat(margin=sys.maxsize).strip()
                    while re.search("\(([A-Z0-9]{1,})((-|=)[A-Z0-9]*)*\s{1,}\)", out) is not None:
                        out = re.sub("\(([A-Z0-9]{1,})((-|=)[A-Z0-9]*)*\s{1,}\)", "", out)
                    out = out.replace(" )", ")")
                    out = re.sub("\s{2,}", " ", out)

                f_out.write(out + "\n")
        f_out.close()

    save_file(train_file_ids, PTB_TRAIN_GOLD_WITHOUT_PUNCTUATION_PATH, include_punctuation=False)
    save_file(val_file_ids, PTB_VALID_GOLD_WITHOUT_PUNCTUATION_PATH, include_punctuation=False)
    save_file(test_file_ids, PTB_TEST_GOLD_WITHOUT_PUNCTUATION_PATH, include_punctuation=False)

    # Align PTB with Yoon Kim's row order
    ptb_train_index_mapper = AlignPTBYoonKimFormat(
        ptb_data_path=PTB_TRAIN_GOLD_WITHOUT_PUNCTUATION_PATH, yk_data_path=YOON_KIM_TRAIN_GOLD_WITHOUT_PUNCTUATION_PATH
    ).row_mapper(save_data_path=PTB_TRAIN_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH)
    ptb_valid_index_mapper = AlignPTBYoonKimFormat(
        ptb_data_path=PTB_VALID_GOLD_WITHOUT_PUNCTUATION_PATH, yk_data_path=YOON_KIM_VALID_GOLD_WITHOUT_PUNCTUATION_PATH
    ).row_mapper(save_data_path=PTB_VALID_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH)
    ptb_test_index_mapper = AlignPTBYoonKimFormat(
        ptb_data_path=PTB_TEST_GOLD_WITHOUT_PUNCTUATION_PATH, yk_data_path=YOON_KIM_TEST_GOLD_WITHOUT_PUNCTUATION_PATH
    ).row_mapper(save_data_path=PTB_TEST_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH)

    # Extract sentences without punctuation
    ptb_train_without_punctuation = pd.read_csv(PTB_TRAIN_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH, sep="\t", header=None, names=["tree"])
    ptb_train_without_punctuation["tree"].apply(extract_sentence).to_csv(
        PTB_TRAIN_SENTENCES_WITHOUT_PUNCTUATION_PATH, index=False, sep="\t", header=None
    )
    ptb_valid_without_punctuation = pd.read_csv(PTB_VALID_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH, sep="\t", header=None, names=["tree"])
    ptb_valid_without_punctuation["tree"].apply(extract_sentence).to_csv(
        PTB_VALID_SENTENCES_WITHOUT_PUNCTUATION_PATH, index=False, sep="\t", header=None
    )
    ptb_test_without_punctuation = pd.read_csv(PTB_TEST_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH, sep="\t", header=None, names=["tree"])
    ptb_test_without_punctuation["tree"].apply(extract_sentence).to_csv(
        PTB_TEST_SENTENCES_WITHOUT_PUNCTUATION_PATH, index=False, sep="\t", header=None
    )

    save_file(train_file_ids, PTB_TRAIN_SENTENCES_WITH_PUNCTUATION_PATH, include_punctuation=True)
    save_file(val_file_ids, PTB_VALID_SENTENCES_WITH_PUNCTUATION_PATH, include_punctuation=True)
    save_file(test_file_ids, PTB_TEST_SENTENCES_WITH_PUNCTUATION_PATH, include_punctuation=True)

    # Extract sentences with punctuation
    ptb_train_with_punctuation = pd.read_csv(PTB_TRAIN_SENTENCES_WITH_PUNCTUATION_PATH, sep="\t", header=None, names=["sentence"])
    ptb_train_with_punctuation = ptb_train_with_punctuation.loc[ptb_train_with_punctuation.index.map(ptb_train_index_mapper)]
    ptb_train_with_punctuation.to_csv(PTB_TRAIN_SENTENCES_WITH_PUNCTUATION_PATH, index=False, sep="\t", header=None)
    ptb_valid_with_punctuation = pd.read_csv(PTB_VALID_SENTENCES_WITH_PUNCTUATION_PATH, sep="\t", header=None, names=["sentence"])
    ptb_valid_with_punctuation = ptb_valid_with_punctuation.loc[ptb_valid_with_punctuation.index.map(ptb_valid_index_mapper)]
    ptb_valid_with_punctuation.to_csv(PTB_VALID_SENTENCES_WITH_PUNCTUATION_PATH, index=False, sep="\t", header=None)
    ptb_test_with_punctuation = pd.read_csv(PTB_TEST_SENTENCES_WITH_PUNCTUATION_PATH, sep="\t", header=None, names=["sentence"])
    ptb_test_with_punctuation = ptb_test_with_punctuation.loc[ptb_test_with_punctuation.index.map(ptb_test_index_mapper)]
    ptb_test_with_punctuation.to_csv(PTB_TEST_SENTENCES_WITH_PUNCTUATION_PATH, index=False, sep="\t", header=None)


def main(arguments):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ptb_path", help="Path to parsed/mrg/wsj folder", type=str, default="./TEMP/corrected/parsed/mrg/wsj/")
    parser.add_argument("--output_path", help="Path to save processed files", type=str, default="./data/PROCESSED/english/")
    args = parser.parse_args(arguments)
    get_data_ptb(args.ptb_path, args.output_path)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
