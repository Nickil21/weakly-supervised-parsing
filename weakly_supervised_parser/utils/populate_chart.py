import pandas as pd
import numpy as np

import datasets

from weakly_supervised_parser.utils.prepare_dataset import DataLoader
from pytorch_lightning import LightningDataModule, Trainer
from weakly_supervised_parser.model.span_classifier import InsideOutsideStringClassifier, DataModule
from weakly_supervised_parser.utils.prepare_dataset import NGramify
from weakly_supervised_parser.utils.create_inside_outside_strings import InsideOutside

from weakly_supervised_parser.settings import INSIDE_MODEL_PATH



class DataModule(LightningDataModule):
    
    def __init__(self, data):
        self.data = data
        
    def test_dataloader(self):
        return DataLoader(datasets.Dataset.from_pandas(self.data))



class PopulateCKYChart:
    def __init__(self, sentence):
        self.sentence = sentence
        self.sentence_length = len(sentence.split())
        self.span_scores = np.zeros((self.sentence_length + 1, self.sentence_length + 1), dtype=float)
        self.all_spans = NGramify(self.sentence).generate_ngrams(single_span=True, whole_span=True)

    def fill_cell(self, span, eval_data):
        cell_score = eval_data.loc[eval_data["span"] == span, "scores"].item()
        return cell_score
        
    def sanity_check(self, eval_data):
        eval_data.loc[eval_data["inside_sentence"].str.split().str.len() == 1, "scores"] = 1
        eval_data.loc[eval_data["inside_sentence"].str.split().str.len() == self.sentence_length, "scores"] = 1
        eval_data["scores"] = eval_data["scores"].clip(lower=0.0, upper=1.0)
        return eval_data

    def fill_chart(self):
        inside_strings = []
        outside_strings = []

        for span in self.all_spans:
            _, inside_string, outside_string = InsideOutside(sentence=self.sentence).create_inside_outside_matrix(span)
            inside_strings.append(inside_string)
            outside_strings.append(outside_string)

        data = pd.DataFrame({"inside_sentence": inside_strings, "outside_sentence": outside_strings, "span": [span[0] for span in self.all_spans]})
        
        # setup your datamodule
        test_dm = DataModule(data=data)
        inside_model = InsideOutsideStringClassifier.load_from_checkpoint(checkpoint_path=INSIDE_MODEL_PATH + "best_model.ckpt")
        trainer = Trainer(gpus=1)
        trainer.test(model=inside_model, dataloaders=test_dm)
        
#         data["inside_scores"] = InsideOutsideStringPredictor(eval_dataset=data[["inside_sentence"]], span_method="Inside").predict_batch(inside_model_path=inside_model_path)
        # data["outside_scores"] = InsideOutsideStringPredictor(eval_dataset=data[["outside_sentence"]], span_method="Outside").predict_batch(outside_model_path=outside_model_path)
        data["scores"] = data["inside_scores"].copy()

        data = self.sanity_check(eval_data=data)

        for span in self.all_spans:
            for i in range(0, self.sentence_length):
                for j in range(i + 1, self.sentence_length + 1):
                    if span[0] == (i, j):
                        self.span_scores[i, j] = self.fill_cell(span[0], data)
        return self.span_scores
