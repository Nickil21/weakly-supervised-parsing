import torch
import datasets
import torchmetrics
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule, LightningModule
from transformers import AdamW, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from onnxruntime import InferenceSession
from scipy.special import softmax


class DataModule(LightningDataModule):

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
        self,
        model_name_or_path: str,
        num_labels=2,
        max_seq_length=192,
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=16,
        train_data_path=None,
        validation_data_path=None,
        **kwargs
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.train_data_path = train_data_path
        self.validation_data_path = validation_data_path
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers

        self.text_fields = "sentence"
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, stage: str):

        if self.train_data_path and self.validation_data_path:
            self.dataset = datasets.load_dataset("csv", data_files={"train": self.train_data_path, "validation": self.validation_data_path})
            for split in self.dataset.keys():
                self.dataset[split] = self.dataset[split].map(
                    self.convert_to_features,
                    batched=True,
                    remove_columns=["label"],
                )
                self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
                self.dataset[split].set_format(type="torch", columns=self.columns)

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size, num_workers=self.num_workers, pin_memory=True)

    def convert_to_features(self, example_batch, indices=None):

        texts_or_text_pairs = example_batch[self.text_fields]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs, max_length=self.max_seq_length, padding="max_length", add_special_tokens=True, truncation=True
        )

        if "label" in example_batch:
            # Rename label to labels to make it easier to pass to model forward
            features["labels"] = example_batch["label"]

        return features


class InsideOutsideStringClassifier(LightningModule):
    def __init__(
        self, model_name_or_path: str, num_labels=2, lr=3e-6, train_batch_size=2, adam_epsilon=1e-8, warmup_steps=0, weight_decay=0.0, **kwargs
    ):
        super().__init__()

        self.save_hyperparameters()

        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=self.num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.model.gradient_checkpointing_enable()
        self.lr = lr
        self.train_batch_size = train_batch_size
        self.accuracy = torchmetrics.Accuracy()
        self.f1score = torchmetrics.F1Score(num_classes=2, multiclass=True)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        val_loss, logits = outputs[:2]
        preds = torch.argmax(logits, axis=1)
        labels = batch["labels"]
        return {"loss": val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", self.accuracy(torch.from_numpy(preds).float().cuda(), torch.from_numpy(labels).cuda()), prog_bar=True)
        self.log("val_f1", self.f1score(torch.from_numpy(preds).float().cuda(), torch.from_numpy(labels).cuda()), prog_bar=True)
        return loss

    def setup(self, stage=None) -> None:
        if stage != "fit":
            return
        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        train_loader = self.trainer.datamodule.train_dataloader()

        # Calculate total steps
        tb_size = self.train_batch_size * max(1, self.trainer.gpus)
        ab_size = tb_size * self.trainer.accumulate_grad_batches
        self.total_steps = int((len(train_loader.dataset) / ab_size) * float(self.trainer.max_epochs))

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.lr,
            eps=self.hparams.adam_epsilon,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


class InsideOutsideStringPredictor:
    def __init__(self, model_name_or_path, pre_trained_model_path, max_length):
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        self.pre_trained_model = InferenceSession(pre_trained_model_path, providers=["CUDAExecutionProvider"])
        self.max_length = max_length

    def preprocess_function(self, data):
        features = self.tokenizer(data["sentence"], max_length=self.max_length, padding="max_length", add_special_tokens=True, truncation=True)
        return features

    def predict_span(self, spans):
        spans_dataset = datasets.Dataset.from_pandas(spans)
        processed = spans_dataset.map(self.preprocess_function, batched=True, batch_size=None)

        inputs = {"input": processed["input_ids"], "attention_mask": processed["attention_mask"]}

        out = self.pre_trained_model.run(None, inputs)
        return softmax(out[0])[:, 1]
