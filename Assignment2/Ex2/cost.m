% cost formula
function r = cost(q,d)
r = 50 + (350/(d^0.4)) + 40 * q^(1/2) + 40 * cos(0.02 * pi * q);
end
