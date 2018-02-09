function WS = gradientdesc(Y, W)
  W0 = W(1,1);
  W1 = W(1,2);
  N = size(Y,1);
  pdW0=((N*exp(W0))/(exp(W0)+exp(W1)))-sum(Y(:,1));
  pdW1=((N*exp(W1))/(exp(W0)+exp(W1)))-sum(Y(:,2));

  %W0 = W0-alpha*(pdW0);
  %W1 = W1-alpha*(pdW1);
  
  WS=[pdW0 pdW1];
 endfunction