W=[1 -1];
J_history = zeros(1, 1);
WS_hist=zeros(1, 2);
rates_hist=zeros(1,2);
diff = 0;
J_history(1)=costfunc(W);
WS_hist(1,:)=W;
i=2;
grads=zeros(1,size(W,2));
grads=gradient(W)';

do
  if(J_history(i-1) == 0)
    break;
  endif
  %alpha = (i+10000).^(-1); %Monro Learning Schedule
  ada_r = 0.06./sqrt((10^-9)+sum(grads'.^2, 2));
  %AdaGrad Learning
  currgrad = gradient(W);
  grads = [grads; currgrad'];
  W=W-(ada_r'.*currgrad');
  %W = W - alpha*currgrad';
  J_history = [J_history;costfunc(W)];
  WS_hist = [WS_hist; W];
  rates_hist = [rates_hist; ada_r'];
  costfunc(W)
  %%save Jhist.m J_history
  %diff=(J_history(i-1)-J_history(i))*100/J_history(i-1);
  i = i+1;
until(J_history(i-1) < 0.001);
i
WS_hist;
costfunc(W)
W
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');
%% ============= Part 4: Visualizing J(W_0, W_1) =============

fprintf('Visualizing J(W_0, W_1) ...\n')

% Grid over which we will calculate J
W0_vals = linspace(-4, 4, 100);
W1_vals = linspace(-4, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(W0_vals), length(W1_vals));

% Fill out J_vals
for i = 1:length(W0_vals)
    for j = 1:length(W1_vals)
	  t = [W0_vals(i); W1_vals(j)];
	  J_vals(i,j) = costfunc(t');
    end
end


% Because of the way meshgrids work in the surf command, we need to
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
surf(W0_vals, W1_vals, J_vals)
%plot(WS_hist(:,1), WS_hist(:,2), '*', 'MarkerSize', 5, 'LineWidth', 2);
xlabel('\W_0'); ylabel('\W_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(W0_vals, W1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\W_0'); ylabel('\W_1');
hold on;
plot(WS_hist(:,1), WS_hist(:,2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);