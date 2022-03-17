from argparse import ArgumentParser

from weakly_supervised_parser.model.span_classifier import InsideOutsideStringPredictor
from weakly_supervised_parser.inference import Predictor
from weakly_supervised_parser.tree.helpers import get_constituents, get_distituents


# outside_model = InsideOutsideStringPredictor(model_name_or_path=args.model_name_or_path, 
#                                                 pre_trained_model_path=args.pre_trained_model_path,
#                                                 max_length=args.max_seq_length)

class CoTrainingAlgorithmPipeline:

    def __init__(self, sentences, num_iterations=2):
        self.sentences = sentences
        self.num_iterations = num_iterations
        self.outside_most_confident_constituents = []
        self.outside_most_confident_distituents = []
        self.inside_most_confident_constituents = []
        self.inside_most_confident_distituents = []
        
    def predict(inside_model, outside_model):
        return Predictor(sentence=inside_sentence).obtain_best_parse(inside_model=inside_model, outside_model=outside_model, return_df=True)
    
    def extract_most_confident_outside_using_inside(self, inside_model):
        inside_model_best_parse, inside_model_df = self.predict(inside_model=inside_model)
        inside_model_pseudo_constituents = inside_model_df[inside_model_df["inside_sentence"].isin(get_constituents(inside_model_best_parse))].copy()
        inside_model_pseudo_distituents = inside_model_df[inside_model_df["inside_sentence"].isin(get_distituents(inside_model_best_parse))].copy()
        outside_most_confident_constituents = inside_model_pseudo_constituents.loc[inside_model_psudeo_constituents["inside_scores"] > constituent_threshold, "outside_sentence"].tolist()
        outside_most_confident_distituents = inside_model_pseudo_distituents.loc[inside_model_psudeo_distituents["inside_scores"] < distituent_threshold, "outside_sentence"].tolist()
        self.outside_model_psudeo_constituents.extend(outside_model_psudeo_constituents)
        self.outside_model_psudeo_distituents.extend(outside_model_psudeo_distituents)
        
    def process(self):
        for sentence in self.sentences:
            self.extract_most_confident_outside_using_inside(inside_model)

    
    
        
        
    def extract_most_confident_outside(self, inside_sentence, inside_model, constituent_threshold=0.95, distituent_threshold=0.05):
        
        inside_model_psudeo_constituents = inside_model_df[inside_model_df["inside_sentence"].isin(get_constituents(inside_model_best_parse))].copy()
        inside_model_psudeo_distituents = inside_model_df[inside_model_df["inside_sentence"].isin(get_distituents(inside_model_best_parse))].copy()
        self.outside_most_confident_constituents = inside_model_psudeo_constituents.loc[inside_model_psudeo_constituents["inside_scores"] > constituent_threshold, "outside_sentence"].tolist()
        self.outside_most_confident_distituents = inside_model_psudeo_distituents.loc[inside_model_psudeo_distituents["inside_scores"] < distituent_threshold, "outside_sentence"].tolist()
        return outside_most_confident_constituents, self.outside_most_confident_distituents 
    
    def extract_most_confident_inside(self, outside_sentence, outside_model, constituent_threshold=0.95, distituent_threshold=0.05):
        outside_model_best_parse, outside_model_df = Predictor(sentence=outside_sentence).obtain_best_parse(outside_model=outside_model, 
                                                                                                         inside_model=None, 
                                                                                                         return_df=True)
        outside_model_psudeo_constituents = outside_model_df[inside_model_df["outside_sentence"].isin(get_constituents(inside_model_best_parse))].copy()
        outside_model_psudeo_distituents = outside_model_df[inside_model_df["outside_sentence"].isin(get_distituents(inside_model_best_parse))].copy()
        self.inside_most_confident_constituents = outside_model_psudeo_constituents.loc[outside_model_psudeo_constituents["outside_scores"] > constituent_threshold, "outside_sentence"].tolist()
        self.inside_most_confident_distituents = outside_model_psudeo_distituents.loc[outside_model_psudeo_distituents["outside_scores"] < distituent_threshold, "inside_sentence"].tolist()
        return self.inside_most_confident_constituents, self.inside_most_confident_distituents
    

if __name__ == "__main__":
    # ------------
    # args
    # ------------
    parser = ArgumentParser(description="Training Pipeline for the Inside Outside String Classifier", add_help=True)
    
    parser.add_argument("--seed", type=int, default=42, help="Training seed")

    parser.add_argument(
        "--model_name_or_path", type=str, default="roberta-base", help="Path to pretrained model or model identifier from huggingface.co/models"
    )
    
    parser.add_argument("--pre_trained_model_path", type=str, required=True,
                       help="Path to the pretrained model")
    
    parser.add_argument("--max_seq_length", default=200, type=int, help="The maximum total input sequence length after tokenization")
    
    args = parser.parse_args()
    
    inside_model = InsideOutsideStringPredictor(model_name_or_path=args.model_name_or_path, 
                                                pre_trained_model_path=args.pre_trained_model_path,
                                                max_length=args.max_seq_length)
    
    cotrainer = CoTrainingAlgorithmPipeline()
    print(cotrainer.extract_most_confident_outside(inside_sentence="The cat sat on the table",
                                                   inside_model=inside_model))
