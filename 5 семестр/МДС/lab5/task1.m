% Setting intervals
t1 = 0:0.01:1; % Interval [0, 1]
t2 = 1:0.01:4; % Interval [1, 4]

% Calculate the function values ​​​​at each interval
x1 = t1 + 1;          % x(t) = t + 1 for t ∈ [0, 1]
x2 = -t2 + 3;         % x(t) = -t + 3 for t ∈ [1, 4]

% Combining intervals and values
t = [t1, t2];         % Whole interval [0, 4]
x = [x1, x2];         % Corresponding Function Values

% Plotting a graph
figure;
plot(t, x, 'b', 'LineWidth', 2);
grid on;
xlabel('t');
ylabel('x(t)');
title('Step method');
