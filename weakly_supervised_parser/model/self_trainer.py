import pandas as pd
import numpy as np

from weakly_supervised_parser.settings import TRAINED_MODEL_PATH


class SelfTrainingClassifier:
    def __init__(self, inside_model, num_iterations: int = 5, prob_threshold: float = 0.99):
        self.inside_model = inside_model
        self.num_iterations = num_iterations
        self.prob_threshold = prob_threshold

    def fit(
        self, train_inside, valid_inside, train_batch_size, eval_batch_size, learning_rate, max_epochs, dataloader_num_workers
    ):  # -1 for unlabeled
        inside_strings = train_inside["sentence"].values
        labels = train_inside["label"].values
        unlabeled_inside_strings = inside_strings[labels == -1]
        labeled_inside_strings = inside_strings[labels != -1]
        labeledy = labels[labels != -1]

        train_df = pd.DataFrame(dict(sentence=labeled_inside_strings, label=labeledy))

        self.inside_model.fit(
            train_df=train_df,
            eval_df=valid_inside,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            dataloader_num_workers=dataloader_num_workers,
            outputdir=TRAINED_MODEL_PATH,
            filename="inside_model_self_trained_base",
        )

        self.inside_model.load_model(pre_trained_model_path=TRAINED_MODEL_PATH + f"inside_model_self_trained_base.onnx")

        unlabeledy = self.inside_model.predict(pd.DataFrame(dict(sentence=unlabeled_inside_strings)))
        unlabeledprob = self.inside_model.predict_proba(pd.DataFrame(dict(sentence=unlabeled_inside_strings)))
        unlabeledy_old = []
        # re-train, labeling unlabeled instances with model predictions, until convergence
        idx = 0
        while (len(unlabeledy_old) == 0 or np.any(unlabeledy != unlabeledy_old)) and idx < self.num_iterations:
            unlabeledy_old = np.copy(unlabeledy)
            uidx = np.where((unlabeledprob[:, 0] > self.prob_threshold) | (unlabeledprob[:, 1] > self.prob_threshold))[0]

            train_df_concat = pd.DataFrame(
                dict(
                    sentence=np.concatenate((labeled_inside_strings, unlabeled_inside_strings[uidx])),
                    label=np.hstack((labeledy, unlabeledy_old[uidx])),
                )
            )

            if idx == self.num_iterations - 1:
                filename = f"inside_model_self_trained"
            else:
                filename = f"inside_model_self_trained_{idx}"

            self.inside_model.fit(
                train_df=train_df_concat,
                eval_df=valid_inside,
                train_batch_size=train_batch_size,
                eval_batch_size=eval_batch_size,
                learning_rate=learning_rate,
                max_epochs=max_epochs,
                dataloader_num_workers=dataloader_num_workers,
                outputdir=TRAINED_MODEL_PATH,
                filename=filename,
            )

            self.inside_model.load_model(pre_trained_model_path=TRAINED_MODEL_PATH + f"{filename}.onnx")

            unlabeledy = self.inside_model.predict(pd.DataFrame(dict(sentence=unlabeled_inside_strings)))
            unlabeledprob = self.inside_model.predict_proba(pd.DataFrame(dict(sentence=unlabeled_inside_strings)))
            idx += 1

        return self
