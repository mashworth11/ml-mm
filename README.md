# Machine learning-based multiscale constitutive modelling

**Authors**: Mark Ashworth, Ahmed Elsheikh, Florian Doster
* Framework for sequential multiscale modelling using machine learning (ML) models to bridge between scales.
* ML model represents a surrogate constitutive model to be used in a physics-based simulator.
* We apply the framework to a (dual-porosity) pressure diffusion problem in the which physical process of interest is traditionally modelled using an (overly-simply) linear relation between pressures.
* Physical process of interest is non-local in time. Accordingly we test various ML methods for modelling the time-dependent problem. 
* We inject the resulting ML model for our time-dependent process into a physics-based simulator. 
* The resulting multiscale approach lets us combine the accuracy of microscale models with the practicality of macroscopic models, without significant addition in computational complexity. 

# Prerequisites and usage 
We use the MATLAB Reservoir Simulator Toolbox ([MRST](https://www.sintef.no/projectweb/mrst/)) for our macroscopic model. To use the ML-based mass transfer constitutive model within MRST you will need the double-porosity module contained in this repo. Having initialised the ML transfer object (see ```DP_ML_transfer.m``` in test cases), the object is then called within ```dual-porosity``` >> ```ad_models``` >> ```equations``` >> ```equationsWaterDP``` at each solver iteration. In terms of dependencies we use MRST 2019b, but this should work with more recent versions provided the dual-porosity module is still knocking around. 

# Citation
.... still writing the paper, watch this space!


