%% 2D dual-porosity model using the machine learning transfer model, in this
% case an autoregressive polynomial regression model. 

mrstModule add ad-blackoil ad-core ad-props dual-porosity


%% Setup Grid
G = cartGrid([1,1], [1,1]);
G = computeGeometry(G);


%% Setup flow params
perm = 10*darcy.*ones(G.cells.num,1);
rock = struct('perm', perm, ...
              'poro', ones(G.cells.num, 1)*0.001); 
          
perm_matrix = 1*milli*darcy.*ones(G.cells.num,1);
rock_matrix = struct('perm', perm_matrix, ...
              'poro', ones(G.cells.num, 1)*0.2); 
          
fluid = initSimpleADIFluid('phases', 'W', 'mu', 1*centi*poise, 'rho', ...
                                   1000*kilogram/meter^3, 'c', ...
                                   4E-10, 'cR', 1E-9, 'pRef', 0);
fluid_matrix = fluid;


%% Gravity
gravity off;


%% Setup model
model = WaterModelDP(G, {rock, rock_matrix},...
                        {fluid, fluid_matrix});
model.dd_transfer_object = DataDrivenTransfer(pr_model);


%% Initialisation and BCs
state0 = initResSol(G, 1E6);
state0.pressure_matrix = ones(G.cells.num,1)*3;
state0.wellSol = initWellSolAD([], model, state0);
model.FacilityModel = FacilityModel(model);
model.validateModel();

% BCs
bc_f = pside([], G, 'WEST', 1E6, 'sat', 1);
bc_f = pside(bc_f, G, 'EAST', 1E6,'sat', 1);
bc_f = fluxside(bc_f, G, 'SOUTH', 0, 'sat', 1);
bc_f = fluxside(bc_f, G, 'NORTH', 0, 'sat', 1);

% initialise ML model
model.dd_transfer_object.ML_inputs = struct('diff_p2', 0, 'dp1', 0,...
                                            'diff_p1', 0, 'dp0', 0,...
                                            'diff_p0', bc_f.value(1)-state0.pressure_matrix);


%% Setup schedule and structures, and simulate
time = 0:0.1:100;
dt = diff(time);
n = length(dt);
states = cell(n, 1);
states = [{state0}; states];

% manual loop
solver = NonLinearSolver('verbose', true);
ML_inputs = model.dd_transfer_object.ML_inputs;
for i = 1:n
    state = solver.solveTimestep(states{i}, dt(i), model, 'bc', bc_f);
    states{i+1} = state;
    % update ML inputs to current state
    ML_inputs.diff_p2 = ML_inputs.diff_p1;
    ML_inputs.dp1 = ML_inputs.dp0;
    ML_inputs.diff_p1 = ML_inputs.diff_p0;
    ML_inputs.dp0 = (states{i+1}.pressure_matrix-states{i}.pressure_matrix)/dt(i);
    ML_inputs.diff_p0 = bc_f.value(1)-states{i+1}.pressure_matrix;
    model.dd_transfer_object.ML_inputs = ML_inputs;
end

% results
DP_pr_p = zeros(size(states));
for i = 1:length(states)
    DP_pr_p(i) = states{i}.pressure_matrix;
end
                    