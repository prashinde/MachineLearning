function cost = costfunc(x)
  dimension = length(x);
	cost = sum(100*(x(2:dimension)-x(1:dimension-1).^2).^2 + (1-x(1:dimension-1)).^2);
endfunction