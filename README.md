# Machine learning-enhanced multiscale modelling: A double-porosity mass transfer example

**Authors**: Mark Ashworth, Ahmed Elsheikh, Florian Doster

In multiscale modelling, multiple models are used simultaneously to describe scale dependent phenomena in a system of interest. Here we introduce a machine learning (ML) enhanced multiscale modelling framework to the problem of inter-porosity mass transfer in dual-porosity materials. Often this mass transfer constitutive relation is presented under the assumption of pseudosteady-state matrix flow. Instead of making any such assumptions, we use a machine learnt constitutive model based on analytical results for the microscale matrix flow problem. To capture time-dependence in the inter-porosity transfer we use a nonlinear autoregressive model with polynomial regression. Finally, we incorporate the ML model into a dual-porosity flow simulator. We compare the performance of the resulting hybrid modelling approach to a microscale explicit model. Our results show good performance of the machine learning-enhanced approach compared to the explicit microscale simulations. Whilst applied to a simple inter-porosity flow problem, the ideas and framework presented in this work are sufficiently general that they could be used in more complex multiscale settings, with different data generation and learning techniques.

# Prerequisites and usage 
We use the MATLAB Reservoir Simulator Toolbox ([MRST](https://www.sintef.no/projectweb/mrst/)) for our macroscopic model. To use the ML-enhanced transfer model within MRST you will need the double-porosity module contained in this repo. Having initialised the ML transfer object (see ```DP_ML_transfer.m``` in test cases), the object is then called within ```dual-porosity``` >> ```ad_models``` >> ```equations``` >> ```equationsWaterDP``` at each solver iteration. 

# Citation



