import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
import seaborn as sns

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def read_true_model_coeff(filename='true_model_coeff'):
    df = pd.read_csv(filename+'.csv')
    return df['alpha'].to_numpy().flatten()[0], df['beta'].to_numpy().flatten()[0], df['sigma'].to_numpy().flatten()[0]

def read_data(filename='regression_data'):
    df = pd.read_csv(filename+'.csv')
    return df['x'].to_numpy().flatten(), df['y'].to_numpy().flatten()

def calc_coeff(x, y):
    ''' According to: https://en.wikipedia.org/wiki/Simple_linear_regression'''
    alpha_hat = np.sum(x * y) / np.sum(x**2)
    beta_hat = y.mean() - alpha_hat * x.mean()
    y_hat = alpha_hat * x + beta_hat
    eps = y - y_hat
    n = len(x)
    sigma_hat = np.sqrt(1/(n - 2) * np.sum(eps**2))
    return alpha_hat, beta_hat, sigma_hat

def calc_coeff_sd(x, y, sigma):
    ''' According to: https://en.wikipedia.org/wiki/Simple_linear_regression'''
    n = len(x)
    Sx = x.sum()
    Sy = y.sum()
    Sxx = np.sum(x**2)
    Syy = np.sum(y**2)
    Sxy = np.sum(x * y)

    alpha_sd = np.sqrt(sigma**2 / np.sum((x - x.mean())**2))
    beta_sd = alpha_sd * np.sqrt(np.sum(x**2) / n)
    rho = (n * Sxy - Sx * Sy) / np.sqrt((n * Sxx - Sx**2) * (n * Syy - Sy**2))

    return alpha_sd, beta_sd, rho


def plot_regression(x, y, alpha_hat, beta_hat, alpha_true, beta_true, n_grid=100):
    x_grid = np.linspace(x.min(), x.max(), n_grid)
    y_hat = beta_hat + alpha_hat * x_grid
    y_true = beta_true + alpha_true * x_grid

    fig = plt.figure()
    plt.scatter(x, y, c= 'red', label='Data')
    plt.plot(x_grid, y_true, color='black', label='True model', linewidth=2, zorder=111)
    plt.plot(x_grid, y_hat, color='blue',label='Regression model', linewidth=2,zorder=111)
    plt.xlabel('Independent variable X', fontsize=16)
    plt.ylabel('Dependent variable Y', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid()


def plot_coeff_PDF(alpha_hat, alpha_sd, beta_hat, beta_sd, rho, n_grid=100):
    fig, ax = plt.subplots(2,2)
    ax[0,0].set_visible(False)

    alpha_grid = np.linspace(alpha_hat - 5 * alpha_sd, alpha_hat + 5 * alpha_sd, n_grid)
    ax[1,0].plot(alpha_grid, norm.pdf(alpha_grid, loc=alpha_hat, scale=alpha_sd))
    ax[1,0]. set_xlabel('Slope', fontsize=16)
    ax[1,0]. set_ylabel('PDF', fontsize=16)

    beta_grid = np.linspace(beta_hat - 5 * beta_sd, beta_hat + 5 * beta_sd, n_grid)
    ax[0,1].plot(beta_grid, norm.pdf(beta_grid, loc=beta_hat, scale=beta_sd))
    ax[0,1]. set_xlabel('Intercept', fontsize=16)
    ax[0,1]. set_ylabel('PDF', fontsize=16)

    mean = np.array([beta_hat, alpha_hat])
    cov = np.array([[beta_sd**2, alpha_sd *beta_sd * rho], [alpha_sd *beta_sd * rho, alpha_sd**2]])
    alpha_mesh, beta_mesh = np.meshgrid(alpha_grid, beta_grid)
    pdf = np.zeros_like(alpha_mesh)
    for i in range(alpha_mesh.shape[0]):
        for j in range(beta_mesh.shape[1]):
            point = np.array([beta_mesh[i, j], alpha_mesh[i, j]])
            pdf[i, j] = multivariate_normal.pdf(point, mean, cov)

    ax[1,1].contourf(beta_mesh, alpha_mesh, pdf)
    ax[1,1]. set_xlabel('Intercept', fontsize=16)
    ax[1,1]. set_ylabel('Slope', fontsize=16)


#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

if __name__=='__main__':

    alpha_true, beta_true, sigma_true = read_true_model_coeff()

    x_data, y_data = read_data()

    alpha_hat, beta_hat, sigma_hat = calc_coeff(x=x_data, y=y_data)

    alpha_sd, beta_sd, rho = calc_coeff_sd(x=x_data, y=y_data, sigma=sigma_hat)

    y_hat = alpha_hat * x_data + beta_hat

    plot_regression(x=x_data, y=y_data, alpha_hat=alpha_hat, beta_hat=beta_hat, alpha_true=alpha_true, beta_true=beta_true)

    plot_coeff_PDF(alpha_hat=alpha_hat, alpha_sd=alpha_sd, beta_hat=beta_hat, beta_sd=beta_sd, rho=rho)


