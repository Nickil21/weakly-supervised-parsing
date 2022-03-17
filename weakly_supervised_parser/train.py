import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from weakly_supervised_parser.model.span_classifier import InsideOutsideStringClassifier, DataModule
from weakly_supervised_parser.utils.prepare_dataset import PTBDataset
from weakly_supervised_parser.settings import PTB_TRAIN_SENTENCES_WITH_PUNCTUATION_PATH
from weakly_supervised_parser.settings import INSIDE_BOOTSTRAPPED_DATASET_PATH, INSIDE_MODEL_PATH


def load_callbacks(filename, output_path):
    callbacks = []
    callbacks.append(EarlyStopping(monitor="val_loss", patience=2, mode="min", check_finite=True))
    callbacks.append(
        ModelCheckpoint(
            monitor="val_loss",
            dirpath=output_path,
            filename=filename,  # + "-{epoch:02d}-{val_loss:.4f}"
            save_top_k=1,
            save_weights_only=True,
            mode="min",
        )
    )
    return callbacks


def cli_main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser(description="Training Pipeline for the Inside Outside String Classifier", add_help=True)

    parser.add_argument("--seed", type=int, default=42, help="Training seed")

    parser.add_argument(
        "--model_name_or_path", type=str, default="roberta-base", help="Path to pretrained model or model identifier from huggingface.co/models"
    )

    parser.add_argument(
        "--train_data_path", type=str, default=INSIDE_BOOTSTRAPPED_DATASET_PATH + "train_seed_bootstrap.csv", help="The csv file containing the training data"
    )

    parser.add_argument(
        "--validation_data_path",
        type=str,
        default=INSIDE_BOOTSTRAPPED_DATASET_PATH + "validation_seed_bootstrap.csv",
        help="The csv file containing the validation data",
    )

    parser.add_argument("--filename", type=str, required=True, help="Path to save the ONNX pre-trained model")

    parser.add_argument("--output_dir", type=str, default=INSIDE_MODEL_PATH, help="Path to the inside/outside model")

    parser.add_argument("--max_epochs", type=int, default=10, help="Limits training to a max number number of epochs")

    parser.add_argument("--lr", type=float, default=3e-6, help="Learning Rate")

    parser.add_argument("--gpus", type=int, default=min(1, torch.cuda.device_count()), help="Number of GPUs to train on")

    parser.add_argument("--devices", type=int, default=1, help="Number of devices to be used by the accelerator for the training strategy")

    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        help="Supports passing different accelerator types ('cpu', 'gpu', 'tpu', 'ipu', 'auto') as well as custom accelerator instances",
    )

    parser.add_argument("--train_batch_size", type=int, default=32, help="Number of training samples in a batch")

    parser.add_argument("--eval_batch_size", type=int, default=32, help="Number of validation samples in a batch")

    parser.add_argument("--num_workers", default=16, type=int, help="Number of workers used in the data loader")

    parser.add_argument("--max_seq_length", default=256, type=int, help="The maximum total input sequence length after tokenization")

    parser.add_argument("--num_labels", default=2, type=int, help="Binary classification, hence two classes")

    args = parser.parse_args()
    args.callbacks = load_callbacks(filename=args.filename, output_path=args.output_dir)

    # -------------------
    # seed bootstrapping
    # -------------------
    ptb = PTBDataset(training_data_path=PTB_TRAIN_SENTENCES_WITH_PUNCTUATION_PATH)
    ptb.train_validation_split(seed=args.seed)

    # ------------
    # data
    # ------------
    data_module = DataModule(
        model_name_or_path=args.model_name_or_path,
        num_labels=args.num_labels,
        max_seq_length=args.max_seq_length,
        train_data_path=args.train_data_path,
        validation_data_path=args.validation_data_path,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )

    data_module.setup("fit")

    # ------------
    # model
    # ------------

    model = InsideOutsideStringClassifier(
        model_name_or_path=args.model_name_or_path, lr=args.lr, num_labels=args.num_labels, train_batch_size=args.train_batch_size
    )

    # ------------
    # training
    # ------------
    seed_everything(args.seed, workers=True)
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, data_module)
    trainer.validate(model, data_module.val_dataloader())

    # --------
    # saving
    # -------
    train_batch = next(iter(data_module.train_dataloader()))

    model.to_onnx(
        file_path="{}/{}.onnx".format(INSIDE_MODEL_PATH, args.filename),
        input_sample=(train_batch["input_ids"].cuda(), train_batch["attention_mask"].cuda()),
        export_params=True,
        opset_version=11,
        input_names=["input", "attention_mask"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "attention_mask": {0: "batch_size"}, "output": {0: "batch_size"}},
    )


if __name__ == "__main__":
    cli_main()

#  python weakly_supervised_parser/train.py --model_name_or_path roberta-base --seed 42 --filename inside_model --max_seq_length 200