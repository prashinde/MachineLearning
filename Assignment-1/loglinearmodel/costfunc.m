function cost = costfunc(Y, W)
  ft = -sum(Y*W');
  st = size(Y,1)*log(exp(W(1,1))+exp(W(1,2)));
  %st=size(Y,1)*log(exp(W(1,1))+exp(W(1,2)))
  cost = ft+st; 
endfunction