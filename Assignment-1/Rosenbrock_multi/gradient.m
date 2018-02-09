function gradient = gradient(x, alpha)
  	dimension = length(x);
    gradient = zeros(dimension, 1);
		gradient(1:dimension-1) = - 400*x(1:dimension-1).*(x(2:dimension)-x(1:dimension-1).^2) - 2*(1-x(1:dimension-1));
		gradient(2:dimension) = gradient(2:dimension) + 200*(x(2:dimension)-x(1:dimension-1).^2)';
    
    %WS = x-alpha*(gradient)';
 endfunction