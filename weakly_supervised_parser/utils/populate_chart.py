import os
import logging
import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from pytorch_lightning import Trainer
from datasets.utils import set_progress_bar_enabled

from weakly_supervised_parser.utils.prepare_dataset import NGramify
from weakly_supervised_parser.utils.create_inside_outside_strings import InsideOutside
from weakly_supervised_parser.model.span_classifier import InsideOutsideStringClassifier, DataModule
from weakly_supervised_parser.settings import INSIDE_MODEL_PATH


# Disable Dataset.map progress bar
set_progress_bar_enabled(False)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

inside_model = InsideOutsideStringClassifier.load_from_checkpoint(checkpoint_path=INSIDE_MODEL_PATH + "inside_model.ckpt")
# outside_model = InsideOutsideStringClassifier.load_from_checkpoint(checkpoint_path=OUTSIDE_MODEL_PATH + "best_model.ckpt")
trainer = Trainer(accelerator="gpu", strategy="ddp", enable_progress_bar=False, gpus=-1) #, profiler="simple")

STOP = set(stopwords.words("english"))
# STOP.update(helper.get_top_tokens(20))


class PopulateCKYChart:
    def __init__(self, sentence):
        self.sentence = sentence
        self.sentence_length = len(sentence.split())
        self.span_scores = np.zeros((self.sentence_length + 1, self.sentence_length + 1), dtype=float)
        self.all_spans = NGramify(self.sentence).generate_ngrams(single_span=True, whole_span=True)

    def fill_cell(self, span, eval_data):
        cell_score = eval_data.loc[eval_data["span"] == span, "scores"].item()
        return cell_score

    def fill_chart(self):
        inside_strings = []
        outside_strings = []

        for span in self.all_spans:
            _, inside_string, outside_string = InsideOutside(sentence=self.sentence).create_inside_outside_matrix(span)
            inside_strings.append(inside_string)
            outside_strings.append(outside_string)

        data = pd.DataFrame({"inside_sentence": inside_strings, "outside_sentence": outside_strings, "span": [span[0] for span in self.all_spans]})
        data["inside_scores"] = trainer.predict(model=inside_model, 
                                                dataloaders=DataModule(model_name_or_path="roberta-base", 
                                                                       test_data=data.rename(columns={"inside_sentence": "sentence"}),
                                                                       num_labels=2,
                                                                       max_seq_length=128,
                                                                       train_batch_size=32,
                                                                       eval_batch_size=32,
                                                                       num_workers=16))[0]
        # data["outside_scores"] = InsideOutsideStringPredictor(eval_dataset=data[["outside_sentence"]], span_method="Outside").predict_batch(outside_model_path=outside_model_path)
        data["scores"] = data["inside_scores"].copy()
        
        data.loc[(data['inside_sentence'].str.lower().str.startswith("the")) &
                 (data['inside_sentence'].str.lower().str.split().str.len() == 2) &
                 (~data['inside_sentence'].str.lower().str.split().str[-1].isin(STOP)), 'scores'] = 1

        for span in self.all_spans:
            for i in range(0, self.sentence_length):
                for j in range(i + 1, self.sentence_length + 1):
                    if span[0] == (i, j):
                        self.span_scores[i, j] = self.fill_cell(span[0], data)
        return self.span_scores
