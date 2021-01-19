function p_m = analyticalPressure1D(perm, poro, mu, c_t, L, time, p_i, p_f)
%
% Function to calculate 1D pressure diffusion solution as per Lim and Aziz
% (1995) and Zhou, et al. (2017). 
%
% Function takes in matrix permeability (perm / md), porosity (poro), total 
% compressibility (c_t / Pa), fluid viscosity (mew / cp), characteristic length 
% between heterogenieties (L / m) and initial and boundary pressures 
% (p_i and p_f resp. / Pa). 
%
% Returns (average) matrix pressure (p_m / Pa)
%

series = 0;
i = 0;
p_dO = 0;
p_d = 1;

while abs(p_d - p_dO) > 0 % run until dimensionless pressure has converged
    p_dO = p_d;
    series = series + ((8/pi^2)*(1/(2*i + 1)^2))*exp(-(((2*i+1)^2)*pi^2*perm*time)/...
                                                        (poro*mu*c_t*L^2));
    p_d = 1 - series;
    i = i + 1;
end

p_m = p_d*(p_f - p_i) + p_i;

end

