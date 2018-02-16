function gradient = gradient(Z)
  	dim = length(Z);
    gradient = zeros(dim, 1);
		gradient(1:dim-1) = - 400*Z(1:dim-1).*(Z(2:dim)-Z(1:dim-1).^2) - 2*(1-Z(1:dim-1));
		gradient(2:dim) = gradient(2:dim) + 200*(Z(2:dim)-Z(1:dim-1).^2);
endfunction
