import torch
import torchmetrics
from torch.optim import AdamW
from pytorch_lightning import LightningModule
from transformers import AutoConfig, AutoModelForSequenceClassification, get_linear_schedule_with_warmup


class LightningModel(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int = 2,
        lr: float = 5e-6,
        train_batch_size: int = 32,
        adam_epsilon=1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        **kwargs
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
        self.f1score = torchmetrics.F1Score(num_classes=2)
        self.mcc = torchmetrics.MatthewsCorrCoef(num_classes=2)

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
        preds = torch.cat([x["preds"] for x in outputs])
        labels = torch.cat([x["labels"] for x in outputs])
        loss = torch.stack([x["loss"] for x in outputs]).mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", self.accuracy(preds, labels.squeeze()), prog_bar=True)
        self.log("val_f1", self.f1score(preds, labels.squeeze()), prog_bar=True)
        self.log("val_mcc", self.mcc(preds, labels.squeeze()), prog_bar=True)
        return loss

    def setup(self, stage=None):
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
