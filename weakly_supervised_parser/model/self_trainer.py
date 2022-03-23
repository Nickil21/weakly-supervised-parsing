import pandas as pd
import numpy as np

from weakly_supervised_parser.tree.helpers import get_constituents, get_distituents
from weakly_supervised_parser.settings import INSIDE_MODEL_PATH
from weakly_supervised_parser.settings import PTB_TRAIN_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH
from weakly_supervised_parser.settings import PTB_TRAIN_SENTENCES_WITH_PUNCTUATION_PATH, PTB_TRAIN_SENTENCES_WITHOUT_PUNCTUATION_PATH
from weakly_supervised_parser.model.trainer import InsideOutsideStringClassifier
from weakly_supervised_parser.utils.prepare_dataset import PTBDataset, DataLoaderHelper
from weakly_supervised_parser.inference import process_test_sample


def prepare_data_self_train(model, threshold=0.99, num_samples=10, num_valid_rows=1000):
    train_sentences = DataLoaderHelper(input_file_object=PTB_TRAIN_SENTENCES_WITHOUT_PUNCTUATION_PATH).read_lines()
    train_gold_file_path = PTB_TRAIN_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH
    lst = []
    for train_index, train_sentence in enumerate(train_sentences):
        best_parse = process_test_sample(train_index, train_sentence, train_gold_file_path, model=inside_model, batch_size=128)
        
        best_parse_get_constituents = get_constituents(best_parse)
        best_parse_get_distituents = get_distituents(best_parse)
        
        constituents_proba = model.predict_proba(pd.DataFrame(dict(sentence=best_parse_get_constituents)))[:, 1]
        distituents_proba = model.predict_proba(pd.DataFrame(dict(sentence=best_parse_get_distituents)))[:, 0]
        
        df_constituents = pd.DataFrame({"sentence": best_parse_get_constituents,
                                        "label": constituents_proba})
        df_distituents = pd.DataFrame({"sentence": best_parse_get_distituents,
                                       "label": distituents_proba})
        
        df_constituents["label"] = np.where(df_constituents["label"] > threshold, 1, -1)
        df_distituents["label"] = np.where(df_distituents["label"] > threshold, 0, -1)
        
        lst.append(pd.concat([df_constituents, df_distituents]))
        if train_index == num_samples:
            break
            
    df_out = pd.concat(lst).sample(frac=1., random_state=42)
    df_out.reset_index(drop=True, inplace=True)
    least_confident_preds_idx = df_out[df_out["label"] == -1].index.values
    most_confident_preds_idx = df_out[df_out["label"].isin([0, 1])].index.values
    valid_idx = most_confident_preds_idx[:num_valid_rows]
    train_idx = most_confident_preds_idx[num_valid_rows:]
    valid_df = df_out.loc[valid_idx]
    train_df = df_out.loc[np.concatenate((train_idx, least_confident_preds_idx))]
    return train_df, valid_df


class SelfTrainingClassifier:
    
    def __init__(self, inside_model, num_iterations=5, prob_threshold=0.99):
        self.inside_model = inside_model
        self.num_iterations = num_iterations
        self.prob_threshold = prob_threshold 
        
    def fit(self, train_inside, valid_inside): # -1 for unlabeled
        inside_strings = train_inside["sentence"].values
        labels = train_inside["label"].values
        unlabeled_inside_strings = inside_strings[labels == -1]
        labeled_inside_strings = inside_strings[labels != -1]
        labeledy = labels[labels != -1]
        
        train_df = pd.DataFrame(dict(sentence=labeled_inside_strings, label=labeledy))
        
        self.inside_model.fit(train_df=train_df,
                              eval_df=valid_inside,
                              train_batch_size=8,
                              eval_batch_size=8,
                              max_epochs=2,
                              outputdir=INSIDE_MODEL_PATH,
                              filename="inside_model_self_train_model_0")
        
        unlabeledy = self.inside_model.predict(pd.DataFrame(dict(sentence=unlabeled_inside_strings)))
        unlabeledprob = self.inside_model.predict_proba(pd.DataFrame(dict(sentence=unlabeled_inside_strings)))
        unlabeledy_old = []
        # re-train, labeling unlabeled instances with model predictions, until convergence
        idx = 0
        while (len(unlabeledy_old) == 0 or np.any(unlabeledy != unlabeledy_old)) and idx < self.num_iterations:
            unlabeledy_old = np.copy(unlabeledy)
            uidx = np.where((unlabeledprob[:, 0] > self.prob_threshold) | (unlabeledprob[:, 1] > self.prob_threshold))[0]
            
            new_train_df = pd.DataFrame(dict(sentence=np.concatenate((labeled_inside_strings, unlabeled_inside_strings[uidx])), 
                                             label=np.hstack((labeledy, unlabeledy_old[uidx]))))
            
            self.inside_model.fit(train_df=new_train_df,
                                  eval_df=valid_inside,
                                  train_batch_size=8,
                                  eval_batch_size=8,
                                  max_epochs=2,
                                  outputdir=INSIDE_MODEL_PATH,
                                  filename=f"inside_self_train_model_{idx}")
                                           
            self.inside_model.load_model(pre_trained_model_path=INSIDE_MODEL_PATH + f"inside_self_train_model_{idx+1}.onnx",               
                                         providers="CUDAExecutionProvider")
            
            unlabeledy = self.inside_model.predict(pd.DataFrame(dict(sentence=unlabeled_inside_strings)))
            unlabeledprob = self.inside_model.predict_proba(pd.DataFrame(dict(sentence=unlabeled_inside_strings)))
            idx += 1
            
        return self

    
if __name__ == "__main__":
    
    ptb = PTBDataset(data_path=PTB_TRAIN_SENTENCES_WITH_PUNCTUATION_PATH)
    train, validation = ptb.train_validation_split(seed=42)
    
    # instantiate
    inside_model = InsideOutsideStringClassifier(model_name_or_path="roberta-base")

#     # train
#     inside_model.fit( train_df=train, 
#                       eval_df=validation, 
#                       train_batch_size=32,
#                       eval_batch_size=32,
#                       max_epochs=10,
#                       use_gpu=True,
#                       outputdir=INSIDE_MODEL_PATH,
#                       filename="inside_model")
    
    # load trained T5 model
    inside_model.load_model(pre_trained_model_path=INSIDE_MODEL_PATH + "inside_model.onnx", providers="CUDAExecutionProvider")
    
    # predict on train
    newtrain, newvalid = prepare_data_self_train(model=inside_model)
    
    clf = SelfTrainingClassifier(inside_model)
    clf.fit(train_inside=newtrain, valid_inside=newvalid)
