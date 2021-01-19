%% Script for results visualisation of the given test problem
times = 0:0.1:100;
analytical = arrayfun(@(t) analyticalPressure2D(0.5*milli*darcy, 0.2,... 
                                                1*centi*poise, 1.4E-9, 1, t, 1,...
                                                1E6), times(2:end));
analytical = [0, analytical];
colourSet = brewermap(11,'RdBu');
figure
loglog(times(2:2:end), SP_av_p(2:2:end) ,...
      'color', 'black', 'LineWidth', 1.5)
hold on
loglog(times(2:2:end), DP_lt_p(2:2:end), ':', ...
       'color', colourSet(10,:), 'LineWidth', 2)
loglog(times(2:2:end), DP_rr_p(2:2:end), '--',...
       'color', colourSet(2,:), 'LineWidth', 2)
loglog(times(2:2:end), analytical(2:2:end),...
       'color', [0.46600,0.67400,0.18800], 'LineWidth', 2)
hold off
ylim([0,2E6])
xlim([1E-1,100])
%title('Average matrix pressure over time', 'FontSize', 16)
xlabel('Time (s)', 'FontSize', 9);
ylabel('$\overline{p}_m$ (Pa)', 'Interpreter','latex', 'FontSize', 9)
ax = gca;
ax.FontSize = 9;
legend('Microscale', 'DC-L',...
       'DC-ML', 'Analytical', ...
       'location', 'southeast')
       