function p_m = analyticalPressure2D(perm, poro, mu, c_t, L, time, p_i, p_f, LT)
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

i = -1;
series0 = 0;
series = 0;
p_dO = 0;
p_d = 1;

% J0 = chebfun(@(x) besselj(0,x),[0 500]);
% r = roots(J0);
% R = (1/sqrt(pi))*L;
%     series = series + (4/r(n+1)^2)*exp(-(r(n+1)^2*perm*time)/...
%                                         (poro*mu*c_t*R^2));    

while abs(p_d - p_dO) > 0 % run until dimensionless pressure has converged
    p_dO = p_d;
    i = i + 1;
    j = 0;
    series = series + (((8/pi^2)^2)*(1/(((2*i+1)^2)*((2*j+1)^2))))...
                        *exp(-((pi^2*perm*time)/(poro*mu*c_t*L^2))*((2*i+1)^2+(2*j+1)^2));
    while abs(series - series0) > 0 % run until inner summation has converged
        j = j + 1;
        series0 = series;
        series = series + (((8/pi^2)^2)*(1/(((2*i+1)^2)*((2*j+1)^2))))...
                            *exp(-((pi^2*perm*time)/(poro*mu*c_t*L^2))*((2*i+1)^2+(2*j+1)^2));
    end
    p_d = 1 - series;
end

p_m = p_d*(p_f - p_i) + p_i;

end

