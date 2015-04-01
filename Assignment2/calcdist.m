function d = calcdist(routes, Dmat);
% D = CALCDIST(ROUTES, DMAT);
%
% Calculates the total distance travelled per route given different routes.
% INPUT : ROUTES - Matrix showing the paths taken in a travel. Every row
%                  corresponds to one possible path and corresponds to the
%                  indices of the nodes visited. It is assumed that travel
%                  starts and ends in node 1. Hence, if every node is to be
%                  visited once, ROUTES should not contain index 1.
%                  E.g. Consider a 3-node problem.
%                       ROUTES(1,:) = [2 3] --> route is 1-2-3-1
%                       ROUTES(2,:) = [3 2] --> route is 1-3-2-1
%                       ROUTES(3,:) = [2 2] --> route is 1-2-1
%
%         DMAT  - The distance matrix indicating the distances between all
%                 nodes, including node 1. DMAT has to be a square and
%                 symmetric matrix.
%
% OUTPUT: D     - Distance vector containing total travelled distance for
%                 each of the paths in ROUTES (one entry for each row).

% (c) Uzay Kaymak, October 2004

% Check number of inputs
if (nargin < 2),
    error('Not enough input arguments.');
end;

% Determine number of nodes
N = size(Dmat,1);

% Determine number of paths
[mr,nr] = size(routes);

% Extract distances to node 1
sd = Dmat(:,1);

% Transform distance matrix into a vector
df = Dmat(:);

% Initialize output vector to distances of the first leg
d = sd(routes(:,1));

% Loop over remaining elements of the paths
for i = 2:nr,
    h = df(N*(routes(:,i-1)-1)+routes(:,i));
    d = d + h;
end;

% Add the final length of the travel to the distance
d = d + sd(routes(:,nr));
