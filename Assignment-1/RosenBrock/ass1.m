arg_list = argv();
Z0=str2num(arg_list{1})
Z1=str2num(arg_list{2})
W=[Z0 Z1];

%Learning Schedule Parameters
LS=arg_list{3};
if(LS == "RM")
	fprintf('Selecting Robbins Monro Schedule\n');
	tao = str2num(arg_list{4})
	k = str2num(arg_list{5})
else
	fprintf('Selecting Ada Grad Schedule\n');
	eeta = str2num(arg_list{4})
	eps = str2num(arg_list{5})
endif

J_history = zeros(1, 1);
WS_hist=zeros(1, 2);
rates_hist=zeros(1,2);
J_history(1)=costfunc(W);
WS_hist(1,:)=W;
i=2;
grads=zeros(1,size(W,2));
grads=gradient(W)';

do
  if(J_history(i-1) == 0)
    break;
  endif

  if(LS == "RM")
    alpha = (i+tao).^(k); %Monro Learning Schedule
  else
    ada_r = eeta./sqrt((eps)+sum(grads'.^2, 2));
  endif

  currgrad = gradient(W);
  grads = [grads; currgrad'];
  if(LS == "RM")
    W = W-alpha*currgrad';
  else
    W = W-(ada_r'.*currgrad');
  endif

  J_history = [J_history;costfunc(W)];
  WS_hist = [WS_hist; W];
  i = i+1;
  if(mod(i, 10000) == 0)
	costfunc(W)
  endif
until(J_history(i-1) < 0.001);
fprintf('Minimum value:');
costfunc(W)
fprintf('Minimum point:\n');
W
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');
