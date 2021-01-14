%% Script for results visualisation of the given test problem
times = cumsum([0,1*hour*ones(1,30)]);
analytical = arrayfun(@(t) analyticalPressure1D(1*milli*darcy, 0.2,... 
                                        1*centi*poise, 1.4E-9, 1, t, 3,...
                                        1E6, false), times(2:end));
analytical = [0, analytical];
colourSet = brewermap(11,'RdBu');
figure
loglog(times(2:2:end), SP_av_p(2:2:end), 'marker', 'o',...
      'color', colourSet(10,:), 'LineWidth', 1, 'MarkerSize', 4)
hold on
loglog(times(2:2:end), DP_lt_p(2:2:end), 'marker', '^',...
       'color', colourSet(2,:), 'LineWidth', 1,  'MarkerSize', 4)
%hold on
loglog(times(2:2:end), DP_nn_p(2:2:end), 'marker', 'd',...
       'color', [0.4660, 0.6740, 0.1880], 'LineWidth', 1,  'MarkerSize', 4)
%loglog(times(2:2:end), analytical(2:2:end),...
%      ':k', 'LineWidth', 1)
hold off
ylim([0,0.5E6])
xlim([1E3,1e5])
%title('Average matrix pressure over time', 'FontSize', 16)
xlabel('Time (s)', 'FontSize', 9);
ylabel('$\overline{p}_m$ (Pa)', 'Interpreter','latex', 'FontSize', 9)
ax = gca;
ax.FontSize = 9;
legend('Microscale', 'DC-L',...
       'DC-ML',...
       'location', 'southeast')
       