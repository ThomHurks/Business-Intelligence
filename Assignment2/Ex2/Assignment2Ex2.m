% assignment 2 - Ex 2

[q,d] = meshgrid(1:200, 8:200);

rows = size(q,1);
cols = size(q,2);

D = zeros(rows, cols);
P = zeros(rows, cols);

for i=1:rows
   for j=1:cols 
      D(i,j) = ed(q(i,j), d(i,j));
      P(i,j) = profit(q(i,j), d(i,j));
   end
end

% plot the environmental damage
% surf(q,d,D)
% xlabel('d'), ylabel('q'), zlabel('Environmental damage');

% plot the profit
% surf(q,d,P)
% xlabel('d'), ylabel('q'), zlabel('Profit');

% xlim([0.1 200])
% ylim([8 200])

% Choose 20 random solutions
rX = [1:20];
rY = [1:20];
Z  = zeros(rows, cols);
for i=1:20
    ri = randi(rows);
    rj = randi(cols);
    rX(i) = D(ri,rj);
    rY(i) = P(ri,rj);
    Z(ri,rj) = 1;
end

% Plot the random solutions in objective function space
% plot(rX, rY)
surf(D,P,Z)
xlabel('damage'), ylabel('profit'), zlabel('random pair');
