# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 11:20:02 2018

@author: kvasnicka
"""
import numpy as np
import pandas as pd
%matplotlib inline
from matplotlib import pyplot as plt
from scipy.stats import uniform
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# %% Generate data
def make_circle_data(n_samples = 100):
    """Creates pandas dataframe with columns u, x, y, such that u ~ U[0,1], x = cos(2*pi*u), y = sin(2*pi*u)."""
    u = uniform.rvs(0,1,n_samples)
    x = np.cos(2*np.pi*u**2)
    y = np.sin(2*np.pi*u**2)
    return pd.DataFrame({'u':u, 'x':x, 'y':y})

# %% Train network
def train_network(data, test_fraction = 0.1):
    """Returns a trained network object"""
    input_cols = ['u']
    output_cols = ['x','y']
    n_rows = len(data)
    n_test = int(n_rows * test_fraction)
    n_train = n_rows - n_test
    X_train = data[input_cols].values[:n_train]
    X_test = data[input_cols].values[n_train:]
    Y_train = data[output_cols].values[:n_train]
    Y_test = data[output_cols].values[n_train:]
    fitter = MLPRegressor(
                hidden_layer_sizes = (60,60),
                activation = 'relu',
                tol = 1.0e-7,
                alpha = 1.0e-4,
                verbose = True
            )
    fitter.fit(X_train, Y_train)
    print('Fit done. Score: {0}'.format(fitter.score(X_test, Y_test)))
    Y_pred = fitter.predict(X_test)
    chi2 = np.sqrt(np.sum((Y_pred - Y_test)**2, axis = 0)/n_test)
    print('Chi2: '.format(chi2))
    return fitter
    
# %% Run
    samples = make_circle_data(10000)
    samples.head()
    fitter = train_network(samples)
    y_pred = fitter.predict(samples.u.values.reshape(-1,1))
    plt.scatter(samples.x,samples.y, s = 0.5)
    plt.scatter(y_pred[:,0], y_pred[:,1], s = 0.5)
    plt.savefig('pictures/circle_generator.png')
