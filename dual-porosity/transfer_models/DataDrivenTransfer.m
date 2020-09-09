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
        ML_inputs
    end
    
    methods
        function dd_object = DataDrivenTransfer(ML_model)
            % Class constructor
            dd_object.ML_model = ML_model;
            dd_object.ML_inputs = struct();
        end
        
        function [Talpha] = calculate_trans_term(dd_object, varargin)
            % Prediction function wrapper
            if isa(dd_object.ML_model, 'network') % NARX net
                input = varargin;
                Talpha = dd_object.ML_model(input, input);
                Talpha = cell2mat(Talpha);
            else
                varargin = cell2mat(varargin);
                input = varargin;
                if isa(dd_object.ML_model, 'SeriesNetwork')... 
                   || isa(dd_object.ML_model, 'DAGNetwork') % Keras model
                    input = input';
                end
                Talpha = predict(dd_object.ML_model, input);
            end
        end
    end
end

