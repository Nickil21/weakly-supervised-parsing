import pandas as pd

from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule


class PyTorchDataModule(Dataset):
    """PyTorch Dataset class"""

    def __init__(self, model_name_or_path: str, data: pd.DataFrame, max_seq_length: int = 256):
        """
        Initiates a PyTorch Dataset Module for input data
        """
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        self.data = data
        self.max_seq_length = max_seq_length

    def __len__(self):
        """returns length of data"""
        return len(self.data)

    def __getitem__(self, index: int):
        """returns dictionary of input tensors to feed into the model"""

        data_row = self.data.iloc[index]
        sentence = data_row["sentence"]

        sentence_encoding = self.tokenizer(
            sentence,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        out = dict(
            sentence=sentence,
            input_ids=sentence_encoding["input_ids"].flatten(),
            attention_mask=sentence_encoding["attention_mask"].flatten(),
            labels=data_row["label"].flatten(),
        )

        return out


class DataModule(LightningDataModule):
    def __init__(
        self,
        model_name_or_path: str,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        max_seq_length: int = 256,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 16,
        **kwargs
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.train_df = train_df
        self.eval_df = eval_df
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):

        self.train_dataset = PyTorchDataModule(self.model_name_or_path, self.train_df, self.max_seq_length)
        self.eval_dataset = PyTorchDataModule(self.model_name_or_path, self.eval_df, self.max_seq_length)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.eval_dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)