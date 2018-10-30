# SVDNoise
This repository contains various pieces of code for the development of SVD noise generator.

The repo starts with tutorial examples on building neural networks using scikit-learn.

## Example 1: Generate random points on a unit circle
For an input u ~ U[0,1], generate random points x, y so that x^2 + y^2 = 1. Of course the analytic solution is x = cos(2pi.u), y = sin(2pi.u), but we want to use a neural network to do the job.

The example is implemented in two files in the code/ subdirectory: a python file circle.py to be run in Spyder or Atom + Hydrogen, and a Jupyter notebook (Circle_generator.ipynb) to be run using Jupyter. 
