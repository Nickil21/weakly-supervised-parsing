import os
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from transformers import AutoTokenizer
from onnxruntime import InferenceSession
from scipy.special import softmax

from weakly_supervised_parser.model.load_dataset import PyTorchDataModule
from weakly_supervised_parser.model.load_dataset import DataModule
from weakly_supervised_parser.model.span_classifier import LightningModel



class InsideOutsideStringClassifier:
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int = 2
        ):

        self.model_name_or_path = model_name_or_path
        self.num_labels = num_labels

    def fit(
        self,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        outputdir: str,
        filename: str,
        use_gpu: bool = True,
        accelerator: str = "auto",
        max_seq_length: int = 256,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        learning_rate: float = 3e-5,
        max_epochs: int = 10,
        dataloader_num_workers: int = 32,
        seed: int = 42
    ):

        data_module = DataModule(
                model_name_or_path=self.model_name_or_path,
                train_df=train_df,
                eval_df=eval_df,
                max_seq_length=max_seq_length,
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
        callbacks.append(ModelCheckpoint(monitor="val_loss", dirpath=outputdir, filename=filename, save_top_k=1, save_weights_only=True, mode="min"))

        gpus = 1 if use_gpu else 0

        trainer = Trainer(accelerator=accelerator, gpus=gpus, max_epochs=max_epochs, callbacks=callbacks)
        trainer.fit(model, data_module)
        trainer.validate(model, data_module.val_dataloader())

        train_batch = next(iter(data_module.train_dataloader()))

        model.to_onnx(
            file_path="{}/{}.onnx".format(outputdir, filename),
            input_sample=(train_batch["input_ids"].cuda(), train_batch["attention_mask"].cuda()),
            export_params=True,
            opset_version=11,
            input_names=["input", "attention_mask"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "attention_mask": {0: "batch_size"}, "output": {0: "batch_size"}})

    def load_model(self, model_name_or_path, pre_trained_model_path, providers):
        self.model = InferenceSession(pre_trained_model_path, providers=[providers])
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    def predict_span(self, spans, max_seq_length):
        spans_dataset = PyTorchDataModule(model_name_or_path="roberta-base", data=spans, max_seq_length=max_seq_length)
        processed = self.tokenizer(spans_dataset["sentence"], max_length=max_seq_length, padding="max_length", add_special_tokens=True, truncation=True)

        inputs = {"input": processed["input_ids"], "attention_mask": processed["attention_mask"]}

        out = self.model.run(None, inputs)
        return softmax(out[0])[:, 1]
