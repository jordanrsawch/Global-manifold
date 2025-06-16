Partially processed protocol and spectral data available at https://doi.org/10.5281/zenodo.15678283.  

*isingFuncts.py*: essential functions for analysis with the Ising function. All functions work for arbitrary numbers of spins. Treatment of two-spin couplings $J$ is currently specialized to cubic 3D grids $(N_x, N_y, N_z)$. Most functions are readily generalized to arbitrary state graphs. Calculation of the derivative of the friction tensor is specialized to the rate law $w_{\mu\nu} = \left[1 + \exp\left\lbrace \beta (V^\mu - V^\nu) \right\rbrace \right]^{-1}$  

*spectrumFuncts.py*: essential functions for producing continuous spectra along protocols. Uses `rateMatrix` and `eqbm` functions from *isingFuncts.py* 

*relaxationMethod.py*: relaxation method for obtaining protocols. Includes specification of 200 control parameter sets. Parameters must be carefully tuned to ensure stability. In some cases, it is helpful to partially relax a lower dimensional control subset, and then lift the vectors to the desired TD manifold with perturbations in co-moving parameters to avoid metastable traps. Should be re-tooled to make use of *isingFuncts.py*, but currently does not depend on other files. 

*exactExcessWork.py*: numerically integrate the master equation to obtain the excess work as a function of protocol durations $\tau$. Uses `setSpinSystem`, `rateMatrix`, `eqbm` and `gradient` functions from *isingFuncts.py* 