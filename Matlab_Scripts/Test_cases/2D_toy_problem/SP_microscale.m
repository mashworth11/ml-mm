%% 2D (single-porosity) microscale model with a left and right hand pressure 
% boundaries, corresponding to a bounding fractures with top and bottom no
% flux boundaries.

mrstModule add ad-blackoil ad-core ad-props
         
         
%% Setup Grid
dx = 1-0.8*cos((0:2/40:2)*pi);
x = (cumsum(dx(1:end)))/(sum(dx(1:end)));
y = (cumsum(dx(1:end)))/(sum(dx(1:end)));
x = [0, x(1:end)]; y = [0, y(1:end)];
G = tensorGrid(x, y);
%plotGrid(G); ax = gca; ax.FontSize = 16;
G = computeGeometry(G);


%% Setup flow params
perm = 0.5*milli*darcy.*ones(G.cells.num,1);
rock = struct('perm', perm, ...
              'poro', ones(G.cells.num, 1)*0.2); 
fluid = initSimpleADIFluid('phases', 'W', 'mu', 1*centi*poise, 'rho', ...
                                   1000*kilogram/meter^3, 'c', ...
                                   4E-10, 'cR', 1E-9, 'pRef', 0);
                               
                            
%% Gravity
gravity off;


%% Setup model
model = WaterModel(G, rock, fluid, 'verbose', true);
model = model.validateModel();


%% Initialisation and BCs
% initial conditions
initState = initResSol(G, 1);
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
SP_av_p = zeros(size(states)); % single porosity (volume) average pressure
for i = 1:length(states)
    SP_av_p(i) = sum(G.cells.volumes.*states{i}.pressure)/(sum(G.cells.volumes));
end


%% Visualise
%plotToolbar(model.G, states, 'outline', true);
%axis off