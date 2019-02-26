
# coding: utf-8

# # Toy neural network example
#
# We train a neural network giving for u in <0,1> a point (x,y) with coordinates
#
# $x = cos(2\pi u)$
#
# $y = sin(2\pi u)$
#
# that is, a point on a unit circle. The network will work as a random "transformer", transforming a random u in <0,1> to a random point on a unit circle.
#
# # Tasks:
# ### 1. Study the code.
# There are several things that can be new:
# - numpy arrays: notice how we can operate on arrays - square them, take a sine or cosine, reshape them
# - pandas dataframes: notice how we can pack several arrays into a data table - a pandas DataFrame.
# - matplotlib plots: notice how simply we plot things, and
# - seaborn plots that add some more plotting functionality.
#
# ### 2. Study the neural network.
# - Training data. To train a neural network, we have to have a proper sample of training data. In this case, it is very simple to generate them. See what happens when you use a smaller or larger training sample.
# - Hyperparameters. We have some freedom in selecting how the network looks (sizes of hidden layers) and how we train it. Look into scikit-learn documentation (google "scikit learn MLPRegressor") and experiment with various combinations of hyperparameters. Optimize training data size and hyperparameters to achieve best precision.
# Rules for hidden layers:
# (i) In theory, a neural network with a single layer can represent any mapping between inputs and outputs (if sufficiently large).
# (ii) If the problem is strongly non-linear, then a two-layer network can be easier to train.
# (iii) More than two hidden layers are usually unnecessary.
#
# ### 3. Investigate the problem.
# We need a surprisingly large network and a lot of training data to make the example work.
# On the other hand, the training is fairly fast.
#
# A. Create a network that will produce density proportional to $u(1-u)$ on the unit circle (rather than uniform)
#
# B. (DONE) Create a network that will produce uniform density on an ellipse.
#
# C. Our network has a single input, u. Try to add some orthogonal polynomials (Legendre or Chebyshev) as additional inputs and examine how such networks train.
#
# D. Also, try to extend the training interval for u from <0,1> to some <-d,1+d>. Error is located mostly near 0 and 1, so let us better teach periodicity to the network.
#
# E. Optimize network parameters and plot optimization results.

# ## 0. Includes
# Note:
# We can select an alias for the namespace of a particular packages using the "as" clause. However, some aliases are used generally, np for numpy, pd for pandas, plt for matplotlib.pyplot, sns for seaborn. It is not wise to change that.

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import uniform
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
# seaborn
import seaborn as sns


# ## 1. Generate data

# In[2]:


def make_circle_data(n_samples = 100, overlap = 0.0):
    """Creates pandas dataframe with columns u, x, y, such that u ~ U[0,1], x = cos(2*pi*u), y = sin(2*pi*u)."""
    u = uniform.rvs(-overlap,1+overlap,n_samples)
    x = np.cos(2*np.pi*u)
    y = np.sin(2*np.pi*u)
    return pd.DataFrame({'u':u, 'x':x, 'y':y})


# ## 2. Train a network regressor

# In[3]:


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
                alpha = 1.0e-5,
                verbose = False
            )
    fitter.fit(X_train, Y_train)
    print('Fit done. Score: {0}'.format(fitter.score(X_test, Y_test)))
    Y_pred = fitter.predict(X_test)
    rmse = np.sqrt(np.sum((Y_pred - Y_test)**2, axis = 0)/n_test)
    print('RMS error: x {0}, y {1}'.format(rmse[0], rmse[1]))
    return fitter


# ## 2. Tune network regressor

# In[4]:


def tune_network(data, test_fraction = 0.1):
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
    fitter0 = MLPRegressor(
                hidden_layer_sizes = (60,60),
                activation = 'logistic',
                tol = 1.0e-7,
                alpha = 1.0e-5,
                verbose = False
            )
    # Optimize for alpha, layer size
    fitter = GridSearchCV(
        fitter0,
        #scoring = 'neg_mean_squared_error',
        param_grid = {
            'alpha': [5.0e-5, 2.0e-5, 1.0e-4, 5.0e-4, 2.0e-4, 1.0e-3],
            'hidden_layer_sizes': [(a, a, a) for a in [20, 25, 30, 35, 40]]
        },
        n_jobs = 4,
        verbose = 1,
    )
    fitter.fit(X_train, Y_train)
    print('Fit done. Score: {0}'.format(fitter.score(X_test, Y_test)))
    Y_pred = fitter.predict(X_test)
    rmse = np.sqrt(np.sum((Y_pred - Y_test)**2, axis = 0)/n_test)
    print('RMS error: x {0}, y {1}'.format(rmse[0], rmse[1]))
    return fitter


# ## 4. Run and plot results

# In[5]:


# Generate samples
samples = make_circle_data(n_samples = 20000)
samples.head()
# Train the network
fitter = tune_network(samples)


# ### Results: x vs. y

# In[6]:


y_pred = fitter.predict(samples.u.values.reshape(-1,1))
fig = plt.figure(figsize = (10,10))
plt.scatter(samples.x,samples.y, s = 0.5)
plt.scatter(y_pred[:,0], y_pred[:,1], s = 0.5)
plt.savefig('../pictures/circle_generator_2d.png')


# ### Results: x, y vs. u

# In[7]:


fig = plt.figure(figsize = (10,5))
plt.scatter(samples.u, samples.x, s = 0.2)
plt.scatter(samples.u, samples.y, s = 0.2)
plt.scatter(samples.u, y_pred[:,0], s = 0.2)
plt.scatter(samples.u, y_pred[:,1], s = 0.2)
plt.savefig('../pictures/circle_generator_1d.png')


# In[8]:


grid_results = pd.DataFrame(fitter.cv_results_)


# In[9]:


grid_results


# ## 5. Save grid search results
# We had best do the plotting in a separate notebook. Grid optimization may take long and we do not want to repeat it if we happen to modify the data.
#
# We do just a single modification to the dataframe before saving: unpack the hidden layer size parameter and rename the column appropriately.

# #### 5.1. Extract layer size from the "param_hidden_layer_sizes" column and rename the column

# In[10]:


grid_results['param_hidden_layer_sizes'] = grid_results.apply(lambda row : row.param_hidden_layer_sizes[0], axis = 1)
grid_results.rename(index = str, columns = {'param_hidden_layer_sizes': 'param_hidden_layer_size'}, inplace = True)
grid_results.head()


# In[11]:


grid_results.to_json('../data/grid_results.json')
