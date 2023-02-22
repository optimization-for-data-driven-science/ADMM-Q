# Alternating Direction Method of Multipliers for Quantization

Quantization of the parameters of machine learning models, such as deep neural networks, requires solving constrained optimization problems, where the constraint set is formed by the Cartesian product of many simple discrete sets. For such optimization problems, we study the performance of the Alternating Direction Method of Multipliers for Quantization (𝙰𝙳𝙼𝙼-𝚀) algorithm, which is a variant of the widely-used ADMM method applied to our discrete optimization problem. We establish the convergence of the iterates of 𝙰𝙳𝙼𝙼-𝚀 to certain stationary points. To the best of our knowledge, this is the first analysis of an ADMM-type method for problems with discrete variables/constraints. Based on our theoretical insights, we develop a few variants of 𝙰𝙳𝙼𝙼-𝚀 that can handle inexact update rules, and have improved performance via the use of "soft projection" and "injecting randomness to the algorithm". 

This repository contains the data, code, and experiments to reproduce our empirical results.


## Getting started

### Dependencies

The following dependencies are needed.
  - Python >= 3.5
  - PyTorch >= 1.1
  - numpy
  - scipy
  - numba
  - xlsxwriter
  - matplotlib

## How to run the code for different examples

**1. MNIST** 
  - We provide "normal train", "binarization" and "binarization_random" functions in the main.py.
  - By default setting, use command `python main.py` will pretrain and then binarize with ADMM-Q.
  - User can modify main function to reproduce other results. For example, at Line 60, change "binarization" to "binarization_random" to test ADMM-R.

**2. Quadratic Example Code**
   - Please do not remove tmp folder
  - d and v can be modified in both run.py and plot.py
  - d is the dimension and v is the variance of \hat{q}. Please refer to the paper for more details. 

  - Run and plot:
  ```
  python run.py
  python plot.py
  ```


  
