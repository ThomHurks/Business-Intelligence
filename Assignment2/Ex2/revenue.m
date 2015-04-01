% revenue formula
function r = revenue(q)
if q >= 10 && q <= 100 
r = 5 * q * atan(220 - q) + 5 * q * cos(0.02 * pi * q);
else
r = 0;
end
