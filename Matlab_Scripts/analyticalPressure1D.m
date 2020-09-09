function p_m = analyticalPressure1D(perm, poro, mu, c_t, L, time, p_i, p_f, LT)
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
n = 0;
p_dO = 0;
p_dN = 1;

if LT == true
    series = series + (8/((2*n + 1)^2*pi^2))*exp(-(((2*n+1)^2)*pi^2*perm*time)/...
                                                      (poro*mu*c_t*L^2));
    p_dN = 1 - series;                                        
else
    while abs(p_dN - p_dO) > 0 
        p_dO = p_dN;
        series = series + (8/((2*n + 1)^2*pi^2))*exp(-(((2*n+1)^2)*pi^2*perm*time)/...
                                                      (poro*mu*c_t*L^2));
        p_dN = 1 - series;
        n = n + 1;
    end
end

p_d = p_dN;
p_m = p_d*(p_f - p_i) + p_i;

end

