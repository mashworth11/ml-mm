%% Script to generate analytical pressure solutions for a range of initial
% pressures
initial_pressures = logspace(0,6,121);
initial_pressures = initial_pressures(1:(end-1));
N = length(initial_pressures);                        
time = [0.1:0.1:100]';
init_pressure = kron(initial_pressures', ones(length(time),1));
fs = array2table(init_pressure, 'VariableNames', cell({'p_i'}));
fs.time = repmat(time, N, 1);

% Paramters
perm = 1*milli*darcy;
poro = 0.2;
mu = 1*centi*poise;
c_t = 1.4E-9;
L = 1;
timescale = (L^2*mu*c_t)/(perm); disp(timescale);
p_f = 1E6;

% Data arrays 
p_m = zeros(height(fs),1);
deriv_p_m = zeros(height(fs),1);
for n = 1:height(fs)
    p_m(n) = analyticalPressure1D(perm, poro, mu, c_t, L, fs.time(n), fs.p_i(n), p_f, false);
end
fs.p_m = p_m;

% write final table to csv
writetable(fs, '../Data/diffusionData.csv')




