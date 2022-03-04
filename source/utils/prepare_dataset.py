from dis import dis
import pandas as pd
from source.utils.process_ptb import punctuation_words, currency_tags_words

class PTBDataset:

    def __init__(self, training_data_path):
        self.data = pd.read_csv(training_data_path, sep="\t", header=None, names=['sentence'])
        self.data['sentence'] = self.data

    def preprocess(self):
        filterchars = punctuation_words + currency_tags_words
        filterchars = [char for char in filterchars if char not in list(",;") and char not in "``" and char not in "\'\'"]
        self.data['sentence'] = self.data['sentence'].apply(lambda row: " ".join([sentence for sentence in row.split() if sentence not in filterchars]))
        return self.data

    def seed_bootstrap_constituent(self):
        constituent_slice_one = self.data['sentence']
        constituent_slice_two = self.data['sentence']
        constituent_samples = pd.DataFrame(dict(sentence=pd.concat([constituent_slice_one, constituent_slice_two]), label=1))
        return constituent_samples

    def seed_bootstrap_distituent(self):
        distituent_slice_one = self.data['sentence'].str.split().str[:-1].str.join(" ")
        distituent_slice_two = self.data[self.data['sentence'].str.split().str.len() > 30]['sentence'].str.split().str[:-2].str.join(" ")
        distituent_slice_three = self.data[self.data['sentence'].str.split().str.len() > 40]['sentence'].str.split().str[:-3].str.join(" ")
        distituent_slice_four = self.data[self.data['sentence'].str.split().str.len() > 50]['sentence'].str.split().str[:-4].str.join(" ")
        distituent_slice_five = self.data[self.data['sentence'].str.split().str.len() > 60]['sentence'].str.split().str[:-5].str.join(" ")
        distituent_slice_six = self.data[self.data['sentence'].str.split().str.len() > 70]['sentence'].str.split().str[:-6].str.join(" ")
        distituent_samples = pd.DataFrame(dict(sentence=pd.concat([distituent_slice_one, distituent_slice_two, 
                                                                   distituent_slice_three, distituent_slice_four,
                                                                   distituent_slice_five, distituent_slice_six]), 
                                               label=0))
        return distituent_samples

    def aggregate_samples(self):
        constituent_samples = self.seed_bootstrap_constituent()
        distituent_samples = self.seed_bootstrap_distituent()
        df = pd.concat([constituent_samples, distituent_samples], ignore_index=True)
        return pd.DataFrame(df.sample(n=200, random_state=42).reset_index(drop=True).to_dict())

    def train_validation_split(self):
        pass


class CTBDataset:

    def __init__(self):
        pass



class KTBDataset:

    def __init__(self):
        pass

    
if __name__ == "__main__":
    ptb = PTBDataset(training_data_path="./data/PROCESSED/english/ptb-train-sentences-with-punctuation.txt")
    # print(ptb.preprocess())
    print(ptb.aggregate_samples().to_csv("sample.csv", index=False))