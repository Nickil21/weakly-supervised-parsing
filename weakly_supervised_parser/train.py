import torch
from argparse import ArgumentParser

from weakly_supervised_parser.settings import TRAINED_MODEL_PATH
from weakly_supervised_parser.utils.prepare_dataset import PTBDataset
from weakly_supervised_parser.model.trainer import InsideOutsideStringClassifier
from weakly_supervised_parser.model.self_trainer import prepare_data_for_self_training, SelfTrainingClassifier
from weakly_supervised_parser.model.co_trainer import prepare_outside_strings, prepare_data_for_co_training, CoTrainingClassifier


def cli_main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser(description="Training Pipeline for the Inside Outside String Classifier", add_help=True)

    parser.add_argument("--seed", type=int, default=42, help="Training seed")

    parser.add_argument(
        "--path_to_train_sentences", type=str, default="roberta-base", help="Path to pretrained model or model identifier from huggingface.co/models"
    )

    parser.add_argument(
        "--model_name_or_path", type=str, default="roberta-base", help="Path to pretrained model or model identifier from huggingface.co/models"
    )

    parser.add_argument("--output_dir", type=str, default=TRAINED_MODEL_PATH, help="Path to the inside/outside model")

    parser.add_argument("--max_epochs", type=int, default=10, help="Limits training to a max number number of epochs")

    parser.add_argument("--lr", type=float, default=5e-6, help="Learning Rate")

    parser.add_argument("--gpus", type=int, default=min(1, torch.cuda.device_count()), help="Number of GPUs to train on")

    parser.add_argument("--strategy", type=str, default="ddp", help="Different training strategies on multiple GPUs")

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

    parser.add_argument("--inside_max_seq_length", default=256, type=int, help="The maximum total input sequence length after tokenization for the inside model")

    parser.add_argument("--outside_max_seq_length", default=64, type=int, help="The maximum total input sequence length after tokenization for the outside model")

    parser.add_argument("--num_labels", default=2, type=int, help="Binary classification, hence two classes")

    parser.add_argument("--num_self_train_iterations", default=5, type=int, help="Number of self-training iterations")

    parser.add_argument("--num_co_train_iterations", default=2, type=int, help="Number of co-training iterations")

    parser.add_argument("--upper_threshold", default=0.99, type=int, help="Threshold value to choose constituents")

    parser.add_argument("--lower_threshold", default=0.01, type=int, help="Threshold value to choose distituents")

    args = parser.parse_args()

    # -------------------
    # seed bootstrapping
    # -------------------
    ptb = PTBDataset(training_data_path=args.path_to_train_sentences)
    train, validation = ptb.train_validation_split(seed=args.seed)

    # -------------------
    # train inside model
    # -------------------
    inside_model = InsideOutsideStringClassifier(model_name_or_path=args.model_name_or_path, 
                                                 max_seq_length=args.inside_max_seq_length)

    print(f"Training the inside model!")
    inside_model.fit(train_df=train, 
                     eval_df=validation,
                     accelerator=args.accelerator,
                     train_batch_size=args.train_batch_size,
                     eval_batch_size=args.eval_batch_size,
                     max_epochs=args.max_epochs,
                     learning_rate=args.lr,
                     use_gpu=args.gpus,
                     dataloader_num_workers=args.num_workers,
                     outputdir=args.output_dir,
                     filename="inside_model")

    # -----------------------------------
    # train inside model w/ self-training
    # -----------------------------------
    inside_model.load_model(pre_trained_model_path=args.output_dir + "inside_model.onnx")
    train_self_trained, valid_self_trained = prepare_data_for_self_training(inside_model=inside_model, train_initial=train, valid_initial=validation, 
                                                                            threshold=args.upper_threshold, num_train_rows=5000, num_valid_examples=1000)

    self_training_clf = SelfTrainingClassifier(inside_model, num_iterations=args.num_self_training_iterations)
    print(f"Training the inside model w/ self-training!")
    self_training_clf.fit(train_inside=train_self_trained, 
                          valid_inside=valid_self_trained,
                          train_batch_size=args.train_batch_size,
                          eval_batch_size=args.eval_batch_size,
                          max_epochs=args.max_epochs,
                          learning_rate=args.lr,
                          dataloader_num_workers=args.num_workers,)

    # --------------------
    # train outside model
    # --------------------
    inside_model.load_model(pre_trained_model_path=args.output_dir + "inside_model_self_trained.onnx")
    outside_model = InsideOutsideStringClassifier(model_name_or_path=args.model_name_or_path, 
                                                  max_seq_length=args.outside_max_seq_length)
    
    train_outside, valid_outside = prepare_outside_strings(inside_model=inside_model, upper_threshold=args.upper_threshold, 
                                                           lower_threshold=args.lower_threshold, num_train_samples=5000, seed=args.seed)

    print(f"Training the outside model!")
    outside_model.fit(train_df=train_outside, 
                      eval_df=valid_outside,
                      accelerator=args.accelerator,
                      train_batch_size=args.train_batch_size,
                      eval_batch_size=args.eval_batch_size,
                      max_epochs=args.max_epochs,
                      learning_rate=args.lr,
                      use_gpu=args.gpus,
                      dataloader_num_workers=args.num_workers,
                      outputdir=args.output_dir,
                      filename="outside_model")

    # ------------------------------------------
    # train inside-outside model w/ co-training
    # ------------------------------------------
    inside_model.load_model(pre_trained_model_path=args.output_dir + "inside_model_self_trained.onnx")
    outside_model.load_model(pre_trained_model_path=args.output_dir + "outside_model.onnx")

    inside_string, outside_string = prepare_data_for_co_training(inside_model, outside_model, upper_threshold=args.upper_threshold, 
                                                                 lower_threshold=args.lower_threshold, seed=args.seed)

    co_training_clf = CoTrainingClassifier(inside_model=inside_model, outside_model=outside_model, 
                                           num_iterations=args.num_co_training_iterations)

    co_training_clf.fit(inside_string=inside_string, outside_string=outside_string)
    