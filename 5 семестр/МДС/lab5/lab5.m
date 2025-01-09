%LabWork 5
close all;

h = 1; %delay
MaxTime = 15;
MinTime = 0;

tspan = [0 MaxTime];
tgrid = linspace(MinTime, MaxTime, 1000);

% Solving
control = false;
sol = dde23(@ddefun, h, @history, tspan);
x = deval(sol, tgrid);

figure
plot(tgrid, x(1,:));
hold on
grid on
plot(tgrid, x(2,:), '--');
legend('x_1','x_2');
xlabel('t');
ylabel('x(t)');



function dxdt = ddefun(t, x, Z)
A = [-5 -2; 1 -2];
A1 = [2 1; 0 -1];

dxdt = zeros(2,1);
dxdt = A*x + A1*Z;
end

function phi = history(t)
phi = zeros(2,1);
phi(1,1) = 5*sin(t);
phi(2,1) = 2*cos(t)+1;
end


