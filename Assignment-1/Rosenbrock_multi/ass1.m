W=rand(1,50,1)
arg_list = argv();

%Learning Schedule Parameters
LS=arg_list{1};
if(LS == "RM")
	fprintf('Selecting Robbins Monro Schedule\n');
	tao = str2num(arg_list{2})
	k = str2num(arg_list{3})
else
	fprintf('Selecting Ada Grad Schedule\n');
	eeta = str2num(arg_list{2})
	eps = str2num(arg_list{3})
endif

J_history = zeros(1, 1);
WS_hist=zeros(1, length(W));
rates_hist=zeros(1,length(W));
J_history(1)=costfunc(W);
WS_hist(1,:)=W;
grads=zeros(1,size(W, 2));
grads=gradient(W)';

i=2;
do
  if(J_history(i-1) == 0)
    break;
  endif
  if(LS == "RM")
    alpha = (i+tao).^(k); %Monro Learning Schedule
  else
    ada_r = eeta./sqrt((eps)+sum(grads'.^2, 2));
  endif

  %AdaGrad Learning
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
  if(mod(i, 100) == 0)
	costfunc(W)
  endif

until(J_history(i-1) < 0.001);
i
fprintf('Minimum value of a function is:');
costfunc(W)
fprintf('Minimum is at a point:');
W
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');
