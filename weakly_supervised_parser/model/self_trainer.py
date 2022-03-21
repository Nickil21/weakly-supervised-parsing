import numpy as np


class SelfTrainingClassifier:
    
    def __init__(self, inside_model, num_iterations=200, prob_threshold=0.8):
        self.inside_model = inside_model
        self.num_iterations = num_iterations
        self.prob_threshold = prob_threshold 
        
    def fit(self, inside_strings, y): # -1 for unlabeled
        unlabeled_inside_strings = inside_strings[y == -1, :]
        labeled_inside_strings = inside_strings[y != -1, :]
        labeledy = y[y != -1]
        
        self.inside_model.fit(labeled_inside_strings, labeledy)
        unlabeledy = self.predict(unlabeled_inside_strings)
        unlabeledprob = self.predict_proba(unlabeled_inside_strings)
        unlabeledy_old = []
        # re-train, labeling unlabeled instances with model predictions, until convergence
        i = 0
        while (len(unlabeledy_old) == 0 or np.any(unlabeledy!=unlabeledy_old)) and i < self.max_iter:
            unlabeledy_old = np.copy(unlabeledy)
            uidx = np.where((unlabeledprob[:, 0] > self.prob_threshold) | (unlabeledprob[:, 1] > self.prob_threshold))[0]
            
            self.inside_model.fit(np.vstack((labeled_inside_strings, unlabeled_inside_strings[uidx, :])), np.hstack((labeledy, unlabeledy_old[uidx])))
            unlabeledy = self.predict(unlabeled_inside_strings)
            unlabeledprob = self.predict_proba(unlabeled_inside_strings)
            i += 1
            
        return self
        
    def predict_proba(self, X):
        return self.inside_model.predict_proba(X)

    def predict(self, X):
        return self.inside_model.predict(X)