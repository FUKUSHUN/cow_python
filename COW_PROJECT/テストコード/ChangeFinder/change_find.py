import numpy as np
import pandas as pd
import changefinder
import hmmlearn.hmm as hmm
import pdb
# my class
import plotting

if __name__ == "__main__":
    filenames = ["./20113.csv", "./20170.csv", "./20295.csv", "./20299.csv"]
    for filename in filenames:
        df = pd.read_csv(filename, usecols=[1,2], names=["time", "score"], header=0)
        ret = []
        model = hmm.GMMHMM(n_components=3, algorithm="viterbi", covariance_type="full", n_iter=100)
        model.fit(np.array([df["score"].values]).T)
        print("means")
        print(model.means_)
        print("covars")
        print(model.covars_)
        ret = pd.Series(model.predict(np.array([df["score"].values]).T), name="state")
        df = pd.concat([df, ret], axis=1)
        # pdb.set_trace()
        df.to_csv(filename)
        
    for filename in filenames:
        df = pd.read_csv(filename, usecols=[1,2,3], names=["time", "score", "state"], header=0)
        plotter = plotting.PlotUtility()
        plotter.scatter_time_plot(df["time"], df["score"], df["state"],size=5)
        plotter.show()
