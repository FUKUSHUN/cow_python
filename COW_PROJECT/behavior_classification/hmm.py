import numpy as np
from hmmlearn import hmm

class hmm_interface:
    model = None
    #emission_matrix = None
    means = None
    covars = None
    transition_matrix = None
    init_matrix = None
        
    def __init__(self, n_state):
        self.model = hmm.GaussianHMM(n_components=n_state, algorithm="viterbi", covariance_type="full")
        #emission_matrix = None
        self.means = None
        self.covars = None
        self.transition_matrix = None
        self.init_matrix = None

    def train_data(self, observation):
        self.model.fit(observation)
        self.means = self.model.means_
        self.covars = self.model.covars_
        self.transition_matrix = self.model.transmat_
        self.init_matrix = self.model.startprob_
        print("遷移行列: ", self.transition_matrix)
        print("出力期待値: ", self.means)
        print("初期確率: ", self.init_matrix)

    def predict_data(self, observation):
        self.model.means_ = self.means
        self.model.covars_ = self.covars
        self.model.transmat_ = self.transition_matrix
        self.model.startprob_ = self.init_matrix
        return self.model.predict(observation)