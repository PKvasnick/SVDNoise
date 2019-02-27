
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

def make_circle_data(n_samples = 100, overlap = 0.0):
    """Creates pandas dataframe with columns u, x, y, such that u ~ U[0,1], x = 2*cos(2*pi*u), y = sin(2*pi*u)."""
    u = uniform.rvs(-overlap,1+overlap,n_samples) 
    x = np.cos(2*np.pi*u)*2
    y = np.sin(2*np.pi*u)
    return pd.DataFrame({'u':u, 'x':x, 'y':y})

# ## 2. Train a network regressor
# Now we tune instead of training

# ## 3. Tune network regressor

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
        #changes in both ranges
        param_grid = {
            'alpha': [1.0e-6, 5.0e-5, 2.0e-5, 1.0e-4, 2.0e-4],
            'hidden_layer_sizes': [(a, a, a) for a in [20, 25, 30, 35, 40, 45]]
           # 'alpha': [5.0e-5, 2.0e-5, 1.0e-4, 5.0e-4, 2.0e-4, 1.0e-3],
           # 'hidden_layer_sizes': [(a, a, a) for a in [20, 25, 30, 35, 40]]
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

# Generate samples
samples = make_circle_data(n_samples = 20000)
samples.head()
# Train the network
fitter = tune_network(samples)

# ### Results: x vs. y

y_pred = fitter.predict(samples.u.values.reshape(-1,1))
fig = plt.figure(figsize = (10,10))
plt.scatter(samples.x,samples.y, s = 0.5)
plt.scatter(y_pred[:,0], y_pred[:,1], s = 0.5)
plt.savefig('../pictures/circle_generator_2d.png')

# ### Results: x, y vs. u

fig = plt.figure(figsize = (10,5))
plt.scatter(samples.u, samples.x, s = 0.2)
plt.scatter(samples.u, samples.y, s = 0.2)
plt.scatter(samples.u, y_pred[:,0], s = 0.2)
plt.scatter(samples.u, y_pred[:,1], s = 0.2)
plt.savefig('../pictures/circle_generator_1d.png')

grid_results = pd.DataFrame(fitter.cv_results_)

#sgrid_results


# ## 5. Save grid search results
# We had best do the plotting in a separate notebook. Grid optimization may take long and we do not want to repeat it if we happen to modify the data. 
# 
# We do just a single modification to the dataframe before saving: unpack the hidden layer size parameter and rename the column appropriately.

# #### 5.1. Extract layer size from the "param_hidden_layer_sizes" column and rename the column

grid_results['param_hidden_layer_sizes'] = grid_results.apply(lambda row : row.param_hidden_layer_sizes[0], axis = 1)
grid_results.rename(index = str, columns = {'param_hidden_layer_sizes': 'param_hidden_layer_size'}, inplace = True)
grid_results.to_json('../data/P_Sgrid_results.json')

# ## 2. Fancy table - seaborn heatmap
# We just cross-tabulate and colour fields according to values.

def make_heatmap(d, xname = 'param_hidden_layer_size', yname = 'param_alpha', valname = '', fmt = 'd'):
    '''Plot a heatmap using columns of dataframe d.'''
    table = pd.pivot_table(d, values = valname, index = yname, columns = xname, aggfunc = np.mean)
    sns.heatmap(table, annot = True, fmt = fmt)
    
make_heatmap(grid_results, valname = 'mean_fit_time', fmt = '.2f')
plt.savefig('../pictures/P_heatmap_mft.png')

# This is good for non-sensitive data, such as training time. For score, we need to see standard deviations, and this we cannot do with this plot.

make_heatmap(grid_results, valname = 'mean_test_score', fmt = '.5f')
plt.savefig('../pictures/P_heatmap_mts.png')

# ## 3. Labelled scatterplot
# 
# Let us try something less fancy, but more useful.

means = ['mean_fit_time', 'mean_test_score']
stds = [s.replace('mean', 'std') for s in means]

grid_results['params'] = grid_results.apply(lambda row : 'alpha: {alpha:.1E}, sizes: {hidden_layer_sizes}'.format(**row.params), axis = 1)

# We sort values by mean score rank
# results['mean_test_score'] = - results['mean_test_score']
grid_results.sort_values(by = ['rank_test_score'], ascending = True, inplace = True)
grid_results

# Subplot frame
top_n = 10 # Only show top 10 results
fig, ax = plt.subplots(nrows = 1, ncols = len(means), sharey = True, figsize = (8,8))
# Individual subplots
for subax, i in zip(ax, range(len(means))):
    xdata = grid_results[means[i]][:top_n]
    ydata = grid_results['rank_test_score'][:top_n]
    xerrors = grid_results[stds[i]][:top_n]
    subax.set_xscale('log', nonposx = 'clip')
    subax.errorbar(x = xdata, y = ydata, xerr = xerrors, fmt = 'o')
    subax.set_yticks(range(1, 1 + len(ydata)))
    subax.set_yticklabels(grid_results['params'])
    subax.grid(color = 'k', linestyle = '-', linewidth = 0.2, axis = 'y')
    subax.set_xlabel(means[i])
plt.savefig('../pictures/P_labelled_scatterplot.png')

