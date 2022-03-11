import torch
import datasets
import torchmetrics
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule, LightningModule
from transformers import AdamW, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


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

    def __init__(self, model_name_or_path, num_labels, max_seq_length, train_batch_size, eval_batch_size, num_workers,
                 train_data_path=None, validation_data_path=None, test_data=None, **kwargs):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.train_data_path = train_data_path
        self.validation_data_path = validation_data_path
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.test_data = test_data

        self.text_fields = ["sentence"]
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, stage: str):
        
        if self.test_data is not None:
            self.test_dataset = datasets.Dataset.from_pandas(self.test_data)
            self.test_dataset = self.test_dataset.map(self.convert_to_features, batched=True, batch_size=None, load_from_cache_file=False)
        
        if self.train_data_path and self.validation_data_path:
            self.dataset = datasets.load_dataset( "csv", data_files={"train": self.train_data_path, "validation": self.validation_data_path})
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

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=len(self.test_dataset), num_workers=self.num_workers, pin_memory=True)

    def convert_to_features(self, example_batch, indices=None):

        texts_or_text_pairs = example_batch[self.text_fields[0]]
        
        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs,
            max_length=self.max_seq_length,
            padding='max_length',
            add_special_tokens=True,
            truncation=True
        )
        
        if "label" in example_batch:
            # Rename label to labels to make it easier to pass to model forward
            features["labels"] = example_batch["label"]
        
        return features
    

class InsideOutsideStringClassifier(LightningModule):
    def __init__(self, model_name_or_path, num_labels, lr, train_batch_size,
                 adam_epsilon=1e-8, warmup_steps=0, weight_decay=0.0, **kwargs):
        super().__init__()

        self.save_hyperparameters()
        
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=self.num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.lr = lr
        self.train_batch_size = train_batch_size
        self.accuracy = torchmetrics.Accuracy()
        self.f1score = torchmetrics.F1Score(num_classes=2, multiclass=True)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        preds = torch.argmax(logits, axis=1)
        labels = batch["labels"]
        return {"loss": val_loss, "preds": preds, "labels": labels}
        
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        input_ids = torch.stack(batch["input_ids"], axis=1)
        attention_mask = torch.stack(batch["attention_mask"], axis=1)
        batch = {"input_ids": input_ids, "attention_mask": attention_mask}
        outputs = self(**batch)
        return torch.nn.functional.softmax(outputs.logits, dim=1)[:, 1]

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
