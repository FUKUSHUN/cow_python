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
        self.model = hmm.GaussianHMM(n_components=n_state, algorithm="viterbi",covariance_type="full")

    def train_data(self, observation):
        self.model.fit(observation)
        self.means = self.model.means_
        self.covars = self.model.covars_
        self.transition_matrix = self.model.transmat_
        self.init_matrix = self.model.startprob_

    def predict_data(self, observation):
        self.model.means_ = self.means
        self.model.covars_ = self.covars
        self.model.transmat_ = self.transition_matrix
        self.model.startprob_ = self.init_matrix
        return self.model.predict(observation)