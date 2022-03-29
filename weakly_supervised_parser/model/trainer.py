import os
import torch
import datasets
import numpy as np
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from transformers import AutoTokenizer

from onnxruntime import InferenceSession
from scipy.special import softmax

from weakly_supervised_parser.model.load_dataset import DataModule
from weakly_supervised_parser.model.span_classifier import LightningModel


class InsideOutsideStringClassifier:
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int = 2,
        max_seq_length: int = 256
        ):

        self.model_name_or_path = model_name_or_path
        self.num_labels = num_labels
        self.max_seq_length = max_seq_length

    def fit(
        self,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        outputdir: str,
        filename: str,
        use_gpu: bool = True,
        accelerator: str = "auto",
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        learning_rate: float = 5e-6,
        max_epochs: int = 10,
        dataloader_num_workers: int = 16,
        seed: int = 42
    ):

        data_module = DataModule(
                model_name_or_path=self.model_name_or_path,
                train_df=train_df,
                eval_df=eval_df,
                max_seq_length=self.max_seq_length,
                train_batch_size=train_batch_size,
                eval_batch_size=eval_batch_size,
                num_workers=dataloader_num_workers
            )
        
        model = LightningModel(model_name_or_path=self.model_name_or_path, 
                              lr=learning_rate, 
                              num_labels=self.num_labels,
                              train_batch_size=train_batch_size,
                              eval_batch_size=eval_batch_size)

        seed_everything(seed, workers=True)
        
        callbacks = []
        callbacks.append(EarlyStopping(monitor="val_loss", patience=2, mode="min", check_finite=True))
        # callbacks.append(ModelCheckpoint(monitor="val_loss", dirpath=outputdir, filename=filename, save_top_k=1, save_weights_only=True, mode="min"))

        gpus = 1 if use_gpu else 0

        trainer = Trainer(accelerator=accelerator, gpus=gpus, max_epochs=max_epochs, callbacks=callbacks)
        trainer.fit(model, data_module)
        trainer.validate(model, data_module.val_dataloader())

        train_batch = next(iter(data_module.train_dataloader()))

        model.to_onnx(
            file_path=f"{outputdir}/{filename}.onnx",
            input_sample=(train_batch["input_ids"].cuda(), train_batch["attention_mask"].cuda()),
            export_params=True,
            opset_version=11,
            input_names=["input", "attention_mask"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "attention_mask": {0: "batch_size"}, "output": {0: "batch_size"}})

    def load_model(self, pre_trained_model_path):
        self.model = InferenceSession(pre_trained_model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def preprocess_function(self, data):
        features = self.tokenizer(data["sentence"],
                                  max_length=self.max_seq_length, 
                                  padding="max_length",
                                  add_special_tokens=True, 
                                  truncation=True, 
                                  return_tensors="np")
        return features
    
    def process_spans(self, spans):
        spans_dataset = datasets.Dataset.from_pandas(spans)
        processed = spans_dataset.map(self.preprocess_function, batched=True, batch_size=None)
        inputs = {"input": processed["input_ids"], "attention_mask": processed["attention_mask"]}
        with torch.no_grad():
            return softmax(self.model.run(None, inputs)[0], axis=1)

    def predict_proba(self, spans, predict_batch_size=128):
        if spans.shape[0] > predict_batch_size:
            output = []
            span_batches = np.array_split(spans, spans.shape[0] // predict_batch_size)
            for span_batch in span_batches:
                output.extend(self.process_spans(span_batch))
            return np.vstack(output)
        else:
            return self.process_spans(spans)
    
    def predict(self, spans):
        return self.predict_proba(spans).argmax(axis=1)
