%% Script for results visualisation of the given test problem
times = 0:0.1:100;
analytical = arrayfun(@(t) analyticalPressure1D(1*milli*darcy, 0.2,... 
                                        1*centi*poise, 1.4E-9, 1, t, 3,...
                                        1E6, false), times(2:end));
analytical = [0, analytical];
colourSet = brewermap(11,'RdBu');
figure
loglog(times(2:2:end), SP_av_p(2:2:end), 'marker', 'x',...
      'color', colourSet(10,:), 'LineWidth', 1, 'MarkerSize', 4)
hold on
loglog(times(2:2:end), DP_lt_p(2:2:end), 'marker', 's',...
       'color', colourSet(2,:), 'LineWidth', 1,  'MarkerSize', 4)
loglog(times(2:2:end), DP_pr_p(2:2:end), 'marker', 'o',...
       'color', [0.4660, 0.6740, 0.1880], 'LineWidth', 1,  'MarkerSize', 4)
loglog(times(2:2:end), analytical(2:2:end),...
      ':k', 'LineWidth', 1)
hold off
ylim([0,1E6])
xlim([1E-1,100])
%title('Average matrix pressure over time', 'FontSize', 16)
xlabel('Time (s)', 'FontSize', 9);
ylabel('$\overline{p}_m$ (Pa)', 'Interpreter','latex', 'FontSize', 9)
ax = gca;
ax.FontSize = 9;
legend('Microscale', 'DP: LT',...
       'DP: MLT', 'Analytical',...
       'location', 'southeast')
       