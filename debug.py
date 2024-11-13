import numpy as np
import pandas as pd
from pathlib import Path

from glmnet import ElasticNet

if __name__ == "__main__":
    
    solve_u_data = Path.home().joinpath("workspace", "pyplier", "tests", "data", "solve_u")

    z_file = solve_u_data / "z.csv.gz"
    z = pd.read_csv(z_file, index_col="gene")

    chat_file = solve_u_data / "chat.csv.gz"
    chat = pd.read_csv(chat_file, index_col="pathway")

    pm_file = solve_u_data / "prior_mat.csv.gz"
    prior_mat = pd.read_csv(pm_file, index_col="gene")
    prior_mat.columns.name = "pathway"

    penalty_factor = np.loadtxt( solve_u_data / "penalty_factor.csv.gz")

    u_file_complete = solve_u_data / "u_complete.csv.gz"
    u_complete = pd.read_csv(u_file_complete, index_col="pathway")

    u_complete.columns = np.subtract(u_complete.columns.str.replace("V", "").astype(int), 1)

    u_complete = u_complete.astype(np.float64)

    u_file_fast = solve_u_data / "u_fast.csv.gz"
    u_fast = pd.read_csv(u_file_fast, index_col="pathway")
    u_fast.columns = np.subtract(u_fast.columns.str.replace("V", "").astype(int), 1).astype(object)
    u_fast = u_fast.astype(np.float64)

    ur = chat @ z  # get U by OLS

    ur = ur.rank(axis="index", ascending=False)  # rank

    iip = np.where([ur.min(axis=1) <= 10])[1]

    results = {}

    u = np.zeros(shape=(prior_mat.shape[1], z.shape[1]))

    lambdas = np.exp(np.arange(start=-4, stop=-12.125, step=-0.125))

    l_mat = np.full((len(lambdas), z.shape[1]), np.nan)

    i = 0
    
    gres = ElasticNet(
        alpha=0.9,
        lower_limits=0,
        lambda_path=lambdas,
        fit_intercept=True,
        standardize=False,
        random_state=0,
        scoring="r2",
        verbose=True,
        max_features=50,
    )
    
    gres.fit(
        X=prior_mat.iloc[:, iip],
        y=z.iloc[:, i],
        relative_penalties=penalty_factor[iip],
    )
    print(gres.coef_)