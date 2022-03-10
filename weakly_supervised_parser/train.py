import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import pandas as pd

import datasets
from pytorch_lightning import Trainer, seed_everything, LightningDataModule, LightningModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from weakly_supervised_parser.model.span_classifier import InsideOutsideStringClassifier, DataModule
from weakly_supervised_parser.utils.prepare_dataset import PTBDataset
from weakly_supervised_parser.settings import PTB_TRAIN_SENTENCES_WITH_PUNCTUATION_PATH
from weakly_supervised_parser.settings import INSIDE_BOOTSTRAPPED_DATASET_PATH, INSIDE_MODEL_PATH

    
if __name__ == "__main__":
    ptb = PTBDataset(training_data_path=PTB_TRAIN_SENTENCES_WITH_PUNCTUATION_PATH)
    ptb.train_validation_split()
    model = InsideOutsideStringClassifier(model_name_or_path="roberta-base", 
                                          train_data_path=INSIDE_BOOTSTRAPPED_DATASET_PATH + "train.csv", 
                                          validation_data_path=INSIDE_BOOTSTRAPPED_DATASET_PATH + "validation.csv",  
                                          save_model_path=INSIDE_MODEL_PATH, 
                                          save_model_filename="inside_model")

    seed_everything(42)
    AVAIL_GPUS = min(1, torch.cuda.device_count())

    dm = DataModule(model_name_or_path="roberta-base",
                    train_data_path=INSIDE_BOOTSTRAPPED_DATASET_PATH + "train.csv", 
                    validation_data_path=INSIDE_BOOTSTRAPPED_DATASET_PATH + "validation.csv")
    dm.setup("fit")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=INSIDE_MODEL_PATH,
        filename="inside_model-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        # every_n_epochs=1,
        mode="min",
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=2, verbose=False, mode="min", check_finite=True)

    trainer = Trainer(accelerator="gpu", strategy="ddp", max_epochs=2, gpus=AVAIL_GPUS, callbacks=[checkpoint_callback, early_stopping])
    trainer.fit(model, datamodule=dm)
    trainer.validate(model, dm.val_dataloader())
