%% 2D dual-porosity model using the (pseudosteady-state) linear transfer 
% model (see Lim and Aziz, 1995). 

mrstModule add ad-blackoil ad-core ad-props dual-porosity


%% Setup Grid
G = cartGrid([1,1], [1,1]);
G = computeGeometry(G);
%plotGrid(G); ax = gca; ax.FontSize = 16;


%% Setup flow params
perm = 10*darcy.*ones(G.cells.num,1);
rock = struct('perm', perm, ...
              'poro', ones(G.cells.num, 1)*0.001); 
          
perm_matrix = 0.5*milli*darcy.*ones(G.cells.num,1);
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
model.transfer_model_object = SimpleTransferFunction();
% shape factor according to Lim and Aziz
L = 1;
a = (pi^2)*(2/(L^2)); % shape factor
model.transfer_model_object.shape_factor_object.shape_factor_value = a;
model.FacilityModel = FacilityModel(model); % needed when using nonlinear solver
model.validateModel();
 

%% Initialisation and BCs
initState = initResSol(G, 1E6);
initState.pressure_matrix = ones(G.cells.num,1)*1;
initState.wellSol = initWellSolAD([], model, initState);

% BCs
bc_f = pside([], G, 'WEST', 1E6, 'sat', 1);
bc_f = pside(bc_f, G, 'EAST', 1E6,'sat', 1);
bc_f = pside(bc_f, G, 'SOUTH', 1E6, 'sat', 1);
bc_f = pside(bc_f, G, 'NORTH', 1E6, 'sat', 1);


%% Setup schedule and simulate
time = 0:0.1:100;
dt = diff(time);

% manual loop
states = cell(length(dt), 1);
states = [{initState}; states];
solver = NonLinearSolver('verbose', true);
for i = 1:length(dt)
    state = solver.solveTimestep(states{i}, dt(i), model, 'bc', bc_f);
    states{i+1} = state;
end

% results
DP_lt_p = zeros(size(states));
for i = 1:length(states)
    DP_lt_p(i) = states{i}.pressure_matrix;
end

DP_lt_pf = zeros(size(states));
for i = 1:length(states)
    DP_lt_pf(i) = states{i}.pressure;
end
                    