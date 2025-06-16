Processed protocol and spectral data available at https://doi.org/10.5281/zenodo.15678283.  

*isingFuncts.py*: essential functions for analysis with the Ising function. Most functions are readily generalized to arbitrary state graphs 

*spectrumFuncts.py*: essential functions for producing continuous spectra along protocols. Uses "rate_matrix" and "eqbm" functions from *isingFuncts.py* 

*relaxationMethod.py*: relaxation method for obtaining protocols. Includes specification of 200 control parameter sets. Parameters must be carefully tuned to ensure stability. In some cases, it is helpful to partially relax a lower dimensional control subset, and then lift the vectors to the desired TD manifold with perturbations in co-moving parameters to avoid metastable traps. Should be re-tooled to make use of *isingFuncts.py*, but currently does not depend on other files. 