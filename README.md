# Machine learning-enhanced multiscale modelling

**Authors**: Mark Ashworth, Ahmed Elsheikh, Florian Doster
* Framework for sequential multiscale modelling using machine learning (ML) models to bridge between scales.
* ML model represents a surrogate constitutive model to be used in a physics-based simulator.
* Apply framework to a (dual-porosity) pressure diffusion problem in which constitutive relations are modelled as (overly-simply) linear relations of the quantities involved.
* Physical process of interest is time-dependent. Accordingly we use autoregressive approaches to model time-dependence, considering various supervised regressors in the process. 
* The resulting multiscale approach lets us combine the accuracy of finescale models with the practicality of coarse models. 

# Prerequisites and usage 
We use the MATLAB Reservoir Simulator Toolbox ([MRST](https://www.sintef.no/projectweb/mrst/)) for our macroscopic model. To use the ML-enhanced mass transfer constitutive model within MRST you will need the double-porosity module contained in this repo. Having initialised the ML transfer object (see ```DP_ML_transfer.m``` in test cases), the object is then called within ```dual-porosity``` >> ```ad_models``` >> ```equations``` >> ```equationsWaterDP``` at each solver iteration. 

# Citation



