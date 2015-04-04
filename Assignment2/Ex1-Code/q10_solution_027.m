ObjectiveFunction = @simple_fitness;
nvars = 2;    % Number of variables
LB = [0 0];   % Lower bound
UB = [1 13];  % Upper bound
% options = gaoptimset('MutationFcn',@mutationadaptfeasible);
options = gaoptimset('CrossoverFcn',@crossoversinglepoint);
options = gaoptimset(options,'PlotFcns',{@gaplotbestf,@gaplotmaxconstr}, ...
    'Display','iter');
% X0 = [0.5 0.5]; % Start point (row vector)
% options = gaoptimset(options,'InitialPopulation',X0);
% Next we run the GA solver.
[x,fval] = ga(ObjectiveFunction,nvars,[],[],[],[],LB,UB, ...
    [],options);

[x_2, fval_2] = fmincon(ObjectiveFunction,x,[],[],[],[]);

options = gaoptimset(options,'StallGenLimit',100);
options = gaoptimset(options,'TolFun',1e-10);

[x_3,fval_3] = ga(ObjectiveFunction,nvars,[],[],[],[],LB,UB, ...
    [],options);