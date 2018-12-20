
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
# We trained the network in the Grid_search_cycle notebook and saved grid optimization data to a json file.
# 
# Here, we will retrieve and analyze the data.
# 
# I will show a zoo of various plot styles using matplotlib and seaborn packages. 
# 
# # Tasks:
# ### 1. Study the code.
# There are several things that can be new:
# - numpy arrays: notice how we can operate on arrays - square them, take a sine or cosine, reshape them
# - pandas dataframes: notice how we can pack several arrays into a data table - a pandas DataFrame.
# - matplotlib plots: notice how simply we plot things, and
# - seaborn plots that add some more plotting functionality.
# 
# ### 2. Analyze the data.
# - Make a comprehensive package of plots and tables that we can use for other similar trainings. 
# 

# ## 0. Includes

# In[1]:


import numpy as np
import pandas as pd
get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
import seaborn as sns


# ## 1. Retrieve data

# In[2]:


results = pd.read_json('../data/grid_results.json')
results.head()


# ## 2. Fancy table - seaborn heatmap
# We just cross-tabulate and colour fields according to values.

# In[3]:


def make_heatmap(d, xname = 'param_hidden_layer_size', yname = 'param_alpha', valname = '', fmt = 'd'):
    '''Plot a heatmap using columns of dataframe d.'''
    table = pd.pivot_table(d, values = valname, index = yname, columns = xname, aggfunc = np.mean)
    sns.heatmap(table, annot = True, fmt = fmt)
    
make_heatmap(results, valname = 'mean_fit_time', fmt = '.2f')


# This is good for non-sensitive data, such as training time. For score, we need to see standard deviations, and this we cannot do with this plot.

# In[4]:


make_heatmap(results, valname = 'mean_test_score', fmt = '.5f')


# ## 3. Labelled scatterplot
# 
# Let us try something less fancy, but more useful.

# In[5]:


means = ['mean_fit_time', 'mean_test_score']
stds = [s.replace('mean', 'std') for s in means]


# In[6]:


results['params'] = results.apply(lambda row : 'alpha: {alpha:.1E}, sizes: {hidden_layer_sizes}'.format(**row.params), axis = 1)


# In[7]:


# We sort values by mean score rank
# results['mean_test_score'] = - results['mean_test_score']
results.sort_values(by = ['rank_test_score'], ascending = True, inplace = True)
results


# In[8]:


# Subplot frame
top_n = 10 # Only show top 10 results
fig, ax = plt.subplots(nrows = 1, ncols = len(means), sharey = True, figsize = (8,8))
# Individual subplots
for subax, i in zip(ax, range(len(means))):
    xdata = results[means[i]][:top_n]
    ydata = results['rank_test_score'][:top_n]
    xerrors = results[stds[i]][:top_n]
    subax.set_xscale('log', nonposx = 'clip')
    subax.errorbar(x = xdata, y = ydata, xerr = xerrors, fmt = 'o')
    subax.set_yticks(range(1, 1 + len(ydata)))
    subax.set_yticklabels(results['params'])
    subax.grid(color = 'k', linestyle = '-', linewidth = 0.2, axis = 'y')
    subax.set_xlabel(means[i])

