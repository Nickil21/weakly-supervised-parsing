# Unsupervised Parsing


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

Unzip the zip file, extract the contents inside the `data/` folder, and delete the original zi Ap file:

    unzip ptb3.zip -d ./TEMP/ && rm ptb3.zip

    cp -r ./TEMP/ ./data/RAW/english/
    mkdir ./data/RAW/english/corpora/ptb/TEMP/
    mv ./data/RAW/english/original/ ./data/RAW/english/corpora/ptb/TEMP/
    mv ./data/RAW/english/corrected/ ./data/RAW/english/corpora/ptb/TEMP/

    python source/utils/process_ptb.py --ptb_path ./TEMP/corrected/parsed/mrg/wsj/ --output_path ./data/PROCESSED/english/
    
Download the files: `ptb-train-gold-filtered.txt`, `ptb-valid-gold-filtered.txt`, and `ptb-test-gold-filtered.txt` from [here](https://drive.google.com/file/d/1m4ssitfkWcDSxAE6UYidrP6TlUctSG2D/view) and place them inside `./data/PROCESSED/english/Yoon_Kim/`.

