import pandas as pd

class AlignPTBYoonKimFormat:

    def __init__(self, ptb_data_path, yk_data_path):
        self.ptb_data = pd.read_csv(ptb_data_path, sep="\t", header=None)
        self.yk_data = pd.read_csv(yk_data_path, sep="\t", header=None)

    def row_mapper(self, save_data_path):
        dict_mapper = self.ptb_data.reset_index().merge(self.yk_data.reset_index(), on=[0]).set_index('index_y')['index_x'].to_dict()
        return self.ptb_data.loc[self.ptb_data.index.map(dict_mapper)].to_csv(save_data_path, sep="\t", index=False, header=None)
