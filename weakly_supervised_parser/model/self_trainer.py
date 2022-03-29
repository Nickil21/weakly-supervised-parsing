import pandas as pd
import numpy as np

from weakly_supervised_parser.tree.helpers import get_constituents, get_distituents
from weakly_supervised_parser.settings import TRAINED_MODEL_PATH
from weakly_supervised_parser.settings import PTB_TRAIN_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH
from weakly_supervised_parser.settings import PTB_TRAIN_SENTENCES_WITHOUT_PUNCTUATION_PATH
from weakly_supervised_parser.utils.prepare_dataset import DataLoaderHelper
from weakly_supervised_parser.inference import process_test_sample


def prepare_data_for_self_training(inside_model, train_initial, valid_initial, threshold, num_train_rows, num_valid_examples, seed):
    train_sentences = DataLoaderHelper(input_file_object=PTB_TRAIN_SENTENCES_WITHOUT_PUNCTUATION_PATH).read_lines()
    train_gold_file_path = PTB_TRAIN_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH
    lst = []
    for train_index, train_sentence in enumerate(train_sentences):
        best_parse = process_test_sample(train_index, train_sentence, train_gold_file_path, model=inside_model)
        
        best_parse_get_constituents = get_constituents(best_parse)
        best_parse_get_distituents = get_distituents(best_parse)
        
        if best_parse_get_constituents:
            constituents_proba = inside_model.predict_proba(pd.DataFrame(dict(sentence=best_parse_get_constituents)))[:, 1]
            df_constituents = pd.DataFrame({"sentence": best_parse_get_constituents, "label": constituents_proba})
            df_constituents["label"] = np.where(df_constituents["label"] > threshold, 1, -1)
        
        if best_parse_get_distituents:
            distituents_proba = inside_model.predict_proba(pd.DataFrame(dict(sentence=best_parse_get_distituents)))[:, 0]
            df_distituents = pd.DataFrame({"sentence": best_parse_get_distituents, "label": distituents_proba})
            df_distituents["label"] = np.where(df_distituents["label"] > threshold, 0, -1)
            
        if best_parse_get_constituents and best_parse_get_distituents:
            out = pd.concat([df_constituents, df_distituents])
            
        elif best_parse_get_constituents and not best_parse_get_distituents:
            out = df_constituents
            
        elif best_parse_get_distituents and not best_parse_get_constituents:
            out = df_distituents
           
        lst.append(out)
        
        if train_index == num_train_rows:
            break
            
    df_out = pd.concat(lst).sample(frac=1., random_state=seed)
    df_out.drop_duplicates(subset=["sentence"], inplace=True)
    df_out.reset_index(drop=True, inplace=True)
    valid_idx = np.concatenate((df_out[df_out["label"] == 1].index.values[:int(num_valid_examples // 4)], 
                                df_out[df_out["label"] == 0].index.values[:int(num_valid_examples // (4/3))]))
    valid_df = df_out.loc[valid_idx]
    train_idx = df_out.loc[~df_out.index.isin(valid_idx)].index.values
    train_df = df_out.loc[np.concatenate((train_idx, df_out[df_out["label"] == -1].index.values))]
    
    train_augmented = pd.concat([train_initial, train_df]).drop_duplicates(subset=["sentence"])
    valid_augmented = pd.concat([valid_initial, valid_df]).drop_duplicates(subset=["sentence"])
    
    return train_augmented, valid_augmented


class SelfTrainingClassifier:
    
    def __init__(
        self, 
        inside_model, 
        num_iterations: int = 5, 
        prob_threshold: float = 0.99
        ):
        self.inside_model = inside_model
        self.num_iterations = num_iterations
        self.prob_threshold = prob_threshold 
        
    def fit(self, train_inside, valid_inside, train_batch_size, eval_batch_size, learning_rate, max_epochs, dataloader_num_workers): # -1 for unlabeled
        inside_strings = train_inside["sentence"].values
        labels = train_inside["label"].values
        unlabeled_inside_strings = inside_strings[labels == -1]
        labeled_inside_strings = inside_strings[labels != -1]
        labeledy = labels[labels != -1]
        
        train_df = pd.DataFrame(dict(sentence=labeled_inside_strings, label=labeledy))
        
        self.inside_model.fit(train_df=train_df,
                              eval_df=valid_inside,
                              train_batch_size=train_batch_size,
                              eval_batch_size=eval_batch_size,
                              learning_rate=learning_rate,
                              max_epochs=max_epochs,
                              dataloader_num_workers=dataloader_num_workers,
                              outputdir=TRAINED_MODEL_PATH,
                              filename="inside_model_self_trained_0")
        
        self.inside_model.load_model(pre_trained_model_path=TRAINED_MODEL_PATH + f"inside_model_self_trained_0.onnx")
        
        unlabeledy = self.inside_model.predict(pd.DataFrame(dict(sentence=unlabeled_inside_strings)))
        unlabeledprob = self.inside_model.predict_proba(pd.DataFrame(dict(sentence=unlabeled_inside_strings)))
        unlabeledy_old = []
        # re-train, labeling unlabeled instances with model predictions, until convergence
        idx = 0
        while (len(unlabeledy_old) == 0 or np.any(unlabeledy != unlabeledy_old)) and idx < self.num_iterations:
            unlabeledy_old = np.copy(unlabeledy)
            uidx = np.where((unlabeledprob[:, 0] > self.prob_threshold) | (unlabeledprob[:, 1] > self.prob_threshold))[0]
            
            train_df_concat = pd.DataFrame(dict(sentence=np.concatenate((labeled_inside_strings, unlabeled_inside_strings[uidx])), 
                                                label=np.hstack((labeledy, unlabeledy_old[uidx]))))

            if idx == self.num_iterations - 1:
                filename = f"inside_model_self_trained"
            else:
                filename = f"inside_model_self_trained_{idx+1}"
            
            self.inside_model.fit(train_df=train_df_concat,
                                  eval_df=valid_inside,
                                  train_batch_size=train_batch_size,
                                  eval_batch_size=eval_batch_size,
                                  learning_rate=learning_rate,
                                  max_epochs=max_epochs,
                                  dataloader_num_workers=dataloader_num_workers,
                                  outputdir=TRAINED_MODEL_PATH,
                                  filename=filename)
                                           
            self.inside_model.load_model(pre_trained_model_path=filename)
            
            unlabeledy = self.inside_model.predict(pd.DataFrame(dict(sentence=unlabeled_inside_strings)))
            unlabeledprob = self.inside_model.predict_proba(pd.DataFrame(dict(sentence=unlabeled_inside_strings)))
            idx += 1
            
        return self
