classdef DataDrivenTransfer
%
% SYNOPSIS:
%   model = DataDrivenTransfer()
%
% DESCRIPTION: 
%   Data driven transfer function class that is instatiated with a machine
%   learned predictor
%
% PARAMETERS:
%   model  - Trained machine learned model 
%
% RETURNS:
%   class instance
%
% EXAMPLE:
%
%
%{
Copyright 2009-2020 SINTEF ICT, Applied Mathematics.

This file is part of The MATLAB Reservoir Simulation Toolbox (MRST).

MRST is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

MRST is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with MRST.  If not, see <http://www.gnu.org/licenses/>.
%}
    
    properties
        ML_model
        compressibility_factor
        ML_inputs
        scaler
    end
    
    methods
        function dd_object = DataDrivenTransfer(ML_model, compressibility_factor)
            % Class constructor
            dd_object.ML_model = ML_model;
            dd_object.compressibility_factor = compressibility_factor;
            dd_object.ML_inputs = struct();
            dd_object.scaler = struct();
        end
        
        function [prediction] = calculate_trans_term(dd_object, varargin)
            % Prediction function wrapper
            if isa(dd_object.ML_model, 'network') % NARX net
                input = varargin;
                prediction = dd_object.ML_model(input, input);
                prediction = cell2mat(prediction);
            else
                varargin = cell2mat(varargin);
                input = varargin;
                if isa(dd_object.ML_model, 'SeriesNetwork')... 
                   || isa(dd_object.ML_model, 'DAGNetwork') % Keras model
                    input = input';
                end
                if isempty(fieldnames(dd_object.scaler)) ~= 1
                    try
                        [mu_x, sigma_x, mu_y, sigma_y] = deal(dd_object.scaler.mu_x,...
                                                              dd_object.scaler.sigma_x,...
                                                              dd_object.scaler.mu_y,...
                                                              dd_object.scaler.sigma_y);
                        input = (input-mu_x)./sigma_x;
                        prediction = predict(dd_object.ML_model, input).*sigma_y + mu_y;
                    catch % DAGNet
                        [mu_y, sigma_y] = deal(dd_object.scaler.mu_y,...
                                               dd_object.scaler.sigma_y);
                        prediction = predict(dd_object.ML_model, input).*sigma_y + mu_y;
                        prediction = prediction';
                    end
                elseif isa(dd_object.ML_model, 'function_handle') % ridge regression model
                    prediction = dd_object.ML_model(input);
                else
                    prediction = predict(dd_object.ML_model, input); % standard regression model
                end
            end
        end
    end
end

