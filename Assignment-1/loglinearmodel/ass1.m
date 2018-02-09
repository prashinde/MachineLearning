N=50;
%Y=rand(N, 1) > 4;
Yt=[0 1 1 1 1 1 0 0 0 0 0 0 1 1 0 1 1 0 0 0 0 1 1 1 0];
Y=Yt';
Y=[~Y Y];
W=[10 -10];
J_history = costfunc(Y, W);;
WS_hist=W;
grads=gradientdesc(Y, W);

do
  oldcost = costfunc(Y, W);
  %alpha = (i+100).^(-0.8); %Monro Learning Schedule
  ada_r = 0.6./sqrt((10^-8)+sum(grads'.^2, 2));
  %AdaGrad Learning
  currgrad=gradientdesc(Y, W);
  grads = [grads; currgrad];
  %W = W-alpha*currgrad;
  W=W-(ada_r'.*currgrad);
  curr_cost = costfunc(Y,W)
  J_history=[J_history; curr_cost];
  WS_hist=[WS_hist; W];
until(oldcost-curr_cost < 0.000001 || curr_cost == 0)
W
costfunc(Y, W)
J_history;
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

%% ============= Part 4: Visualizing J(W_0, W_1) =============
fprintf('Visualizing J(W_0, W_1) ...\n')

% Grid over which we will calculate J
W0_vals = linspace(-10, 10, 100);
W1_vals = linspace(-10, 10, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(W0_vals), length(W1_vals));
ijvals=zeros(length(W0_vals), 2);
% Fill out J_vals
min = 10000000;
I=0;
J=0;
for i = 1:length(W0_vals)
    for j = 1:length(W1_vals)
	  t = [W0_vals(i); W1_vals(j)];
	  J_vals(i,j) = costfunc(Y, t');
    if(J_vals(i,j) < min)
      min = J_vals(i,j);
      I=W0_vals(i);
      J=W1_vals(j);
    endif
    %costfunc(Y,t')
    end
end

min
I
J
% Because of the way meshgrids work in the surf command, we need to
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
surfc(W0_vals, W1_vals, J_vals)
%plot(WS_hist(:,1), WS_hist(:,2), '*', 'MarkerSize', 5, 'LineWidth', 2);
xlabel('\W_0'); ylabel('\W_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(W0_vals, W1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\W_0'); ylabel('\W_1');
hold on;
plot(WS_hist(:,1), WS_hist(:,2), '*', 'MarkerSize', 5, 'LineWidth', 2);
