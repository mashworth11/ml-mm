%% Script to run and visualise the analyticalPressure1D solution

% Setup params
perm = 0.01*milli*darcy;
poro = 0.2;
mu = 1*centi*poise;
c_f =  4E-10; c_r = 1E-9;
c_t = c_f + c_r;
L = 0.001;

% Setup initial and BCs
p_i = 0;
p_f = 3000;

% Time-step solution
time = logspace(-6, 1, 50);
p_d = zeros(length(time), 1);
p_m = zeros(size(p_d));

for i = 1:length(time)
    [p_d(i), p_m(i)] = analyticalPressure1D(perm, poro, mu, c_t, L, time(i), p_i, p_f);
end

% visualise
t_d = (perm/(poro*mu*c_t*L^2))*time;
semilogx(t_d, p_d, 'linewidth', 2)
title('Dimensionless pressure (p_d) vs dimensionless time (t_d)')
xlabel('$t_d = \frac{kt}{\phi\mu c_tL^2}$','interpreter','latex', 'fontsize', 16)
ylabel('$p_d = \frac{\overline{p}_m - p_i}{p_f - p_i}$','interpreter','latex', 'fontsize', 16)

figure 
semilogx(time, p_m, 'linewidth', 2)
title('Average matrix pressure ($\overline{p}_m$) vs time (t)', 'interpreter','latex')
xlabel('$t$', 'interpreter','latex', 'fontsize', 16)
ylabel('$\overline{p}_m$','interpreter','latex', 'fontsize', 16)