<div align="center"> 
    
# Co-training an Unsupervised Constituency Parser with Weak Supervision

[![Conference](http://img.shields.io/badge/ACL%20Findings-2022-ed1c24.svg)](https://arxiv.org/abs/2110.02283)
[![CI testing](https://github.com/Nickil21/weakly-supervised-parsing/actions/workflows/ci-testing.yml/badge.svg)](https://github.com/Nickil21/weakly-supervised-parsing/actions/workflows/ci-testing.yml)
[![Codecov](https://codecov.io/gh/Nickil21/weakly-supervised-parsing/branch/main/graphs/badge.svg?token=8QRSBXTTQX)](https://codecov.io/gh/Nickil21/weakly-supervised-parsing/)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/co-training-an-unsupervised-constituency/constituency-grammar-induction-on-ptb)](https://paperswithcode.com/sota/constituency-grammar-induction-on-ptb?p=co-training-an-unsupervised-constituency)
    
![block diagram](block_diagram.png)

</div>

## Table of Contents

   * [Installing dependencies](#installing-dependencies)
   * [Train](#train)      
   * [Inference](#inference)
   * [Evaluation](#evaluation)
   * [Tests](#tests)

## Installing dependencies

Clone the project:

```shell
git clone https://github.com/Nickil21/weakly-supervised-parsing
cd weakly-supervised-parsing/
```

Create virtual environment:

```shell
python3.8 -m venv weak-sup-parser-env # or other python versions >=3.8
source weak-sup-parser-env/bin/activate
```

Install the requirements

```shell
pip install -r requirements.txt
``` 

## Processing PTB

Set the `NLTK_DATA` variable to allow nltk to find the corpora and resources you downloaded with `nltk.download()`:

```shell
export NLTK_DATA=./data/RAW/english/
python -m nltk.downloader ptb
```

Download the PTB 3.0 file from [LDC99T42](https://catalog.ldc.upenn.edu/LDC99T42) and place it in `./data/RAW/english/corpora/ptb/treebank_3`. After doing this process, `./data/RAW/english/corpora/ptb/treebank_3/parsed/mrg/wsj` should have folders named 00-24.

To process PTB:

```shell
export PYTHONPATH=${PWD}
python weakly_supervised_parser/utils/process_ptb.py
```

Download the following files from [here](https://drive.google.com/file/d/1m4ssitfkWcDSxAE6UYidrP6TlUctSG2D/view) and place them inside `./data/PROCESSED/english/Yoon_Kim/`:

    ptb-train-gold-filtered.txt
    ptb-valid-gold-filtered.txt
    ptb-test-gold-filtered.txt

## Train

```shell
export MODEL_PATH=weakly_supervised_parser/model/TRAINED_MODEL
export TRAIN_SENTENCES_PATH=./data/PROCESSED/english/sentences/ptb-train-sentences-without-punctuation.txt

python weakly_supervised_parser/train.py \
    --path_to_train_sentences ${TRAIN_SENTENCES_PATH} \
    --model_name_or_path roberta-base \
    --output_dir ${MODEL_PATH} \
    --max_epochs 10 \
    --lr 5e-6 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --num_workers 16 \
    --inside_max_seq_length 256 \
    --outside_max_seq_length 64 \
    --num_labels 2 \
    --num_self_train_iterations 5 \
    --num_co_train_iterations 2 \
    --upper_threshold 0.995 \
    --lower_threshold 0.005 \
    --num_train_rows 100 \
    --num_valid_examples 100 \
    --seed 42
```

## Inference

```shell
python weakly_supervised_parser/inference.py \
    --use_inside \
    --model_name_or_path roberta-base \
    --inside_max_seq_length 256 \
    --save_path <FILENAME>
```

## Evaluation

```shell
python weakly_supervised_parser/tree/compare_trees.py --tree2 <FILENAME>             
```

## Tests

```shell
pytest weakly_supervised_parser/tests --disable-pytest-warnings
```
    
## Citation

If you find our paper and code useful in your research, please consider citing:

```bibtex
@inproceedings{maveli-cohen-2022-co,
    title = "{C}o-training an {U}nsupervised {C}onstituency {P}arser with {W}eak {S}upervision",
    author = "Maveli, Nickil  and
      Cohen, Shay",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-acl.101",
    doi = "10.18653/v1/2022.findings-acl.101",
    pages = "1274--1291",
    abstract = "We introduce a method for unsupervised parsing that relies on bootstrapping classifiers to identify if a node dominates a specific span in a sentence. There are two types of classifiers, an inside classifier that acts on a span, and an outside classifier that acts on everything outside of a given span. Through self-training and co-training with the two classifiers, we show that the interplay between them helps improve the accuracy of both, and as a result, effectively parse. A seed bootstrapping technique prepares the data to train these classifiers. Our analyses further validate that such an approach in conjunction with weak supervision using prior branching knowledge of a known language (left/right-branching) and minimal heuristics injects strong inductive bias into the parser, achieving 63.1 F$_1$ on the English (PTB) test set. In addition, we show the effectiveness of our architecture by evaluating on treebanks for Chinese (CTB) and Japanese (KTB) and achieve new state-of-the-art results.",
}
```
