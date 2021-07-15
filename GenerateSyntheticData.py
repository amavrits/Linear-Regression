import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def set_true_linear_model(alpha_true, beta_true):
    return lambda x: beta_true + alpha_true * x

def generate_data(x, true_model, sigma_true):
    y_model = true_model(x)
    y_data = y_model + np.random.randn(len(x)) * sigma_true
    return y_data

def export_true_model_coeff(alpha_true, beta_true, sigma_true, filename='true_model_coeff'):
    df = pd.DataFrame(data=np.c_[alpha_true, beta_true, sigma_true], columns=['alpha', 'beta', 'sigma'])
    df.to_csv(filename+'.csv', index=False)

def export_data(x, y, filename='regression_data'):
    df = pd.DataFrame(data=np.c_[x.reshape(-1,1), y.reshape(-1,1)], columns=['x', 'y'])
    df.to_csv(filename+'.csv', index=False)


#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

if __name__=='__main__':

    alpha_true = 5
    beta_true = 17
    sigma_true = 2

    n_data = 1_000
    x_mean = 100
    x_sd = 5
    x_data = x_mean + np.random.randn(n_data) * x_sd

    f_true = set_true_linear_model(alpha_true=alpha_true, beta_true=beta_true)
    y_data = generate_data(x=x_data, true_model=f_true, sigma_true=sigma_true)
    export_data(x=x_data, y=y_data)
    export_true_model_coeff(alpha_true=alpha_true, beta_true=beta_true, sigma_true=sigma_true)
