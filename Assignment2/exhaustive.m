% all possible routes
apr = perms([2 3 4 5 6 7]);

result = cell(size(apr,1),2);

for r = 1:size(apr,1),
    % Store the distance and the route
    result(r,1) = {calcdist(apr(r,:), Dmatf)};
    result(r,2) = {apr(r,:)};
end;

result = sortrows(result,1);
