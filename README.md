<div align="center"> 
    
# Co-training an Unsupervised Constituency Parser with Weak Supervision

[![Conference](http://img.shields.io/badge/ACL%20Findings-2022-ed1c24.svg)](https://arxiv.org/abs/2110.02283)
[![CI testing](https://github.com/Nickil21/weakly-supervised-parsing/actions/workflows/ci-testing.yml/badge.svg)](https://github.com/Nickil21/weakly-supervised-parsing/actions/workflows/ci-testing.yml)
    
![Block Diagram](https://nickilmaveli.com/assets/images/publications/mscr_thesis.png)

</div>

## Installing Pipenv

Run the following command to ensure you have pip installed in your system:

    pip3 --version

Install pipenv by running the following command:

    sudo -H pip3 install -U pipenv

Activate the Pipenv shell:

    pipenv shell

Install the project dependencies:

    pipenv install 

## Processing PTB

Set the NLTK_DATA variable to allow nltk to find the corpora and resources you downloaded with nltk.download():

    export NLTK_DATA=./data/RAW/english/
    python -m nltk.downloader ptb

Download the PTB 3.0 file:

    wget http://bollin.inf.ed.ac.uk/public/direct/sandbox/ptb3.zip !! TO REMOVE THE LINK LATER

Unzip the zip file, extract the contents inside the `data/` folder, and delete the original zip file:

    unzip ptb3.zip -d ./TEMP/ && rm ptb3.zip

    cp -r ./TEMP/ ./data/RAW/english/
    mkdir ./data/RAW/english/corpora/ptb/TEMP/
    mv ./data/RAW/english/TEMP/corrected/ ./data/RAW/english/corpora/ptb/TEMP/

Download the following files from [here](https://drive.google.com/file/d/1m4ssitfkWcDSxAE6UYidrP6TlUctSG2D/view) and place them inside `./data/PROCESSED/english/Yoon_Kim/`:

    ptb-train-gold-filtered.txt
    ptb-valid-gold-filtered.txt
    ptb-test-gold-filtered.txt

To process PTB:

    python parser/utils/process_ptb.py

Delete the unnecessary PTB files inside the `TEMP/` folder.

    rm -rf ./TEMP/corrected/

## Train

    python weakly_supervised_parser/train.py \
            --model_name_or_path roberta-base \
            --seed 42 \
            --filename inside_model \
            --max_seq_length 256

## Inference

    python weakly_supervised_parser/inference.py --predict_on_test \
            --model_name_or_path roberta-base \
            --pre_trained_model_path weakly_supervised_parser/model/TRAINED_MODEL/INSIDE/inside.onnx \
            --max_seq_length 200 \
            --save_path TEMP/predictions/english/inside_model_predictions.txt

# Tests

    pytest weakly_supervised_parser/tests --disable-pytest-warnings
