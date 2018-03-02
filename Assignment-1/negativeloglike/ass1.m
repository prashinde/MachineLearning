arg_list = argv();
N=str2num(arg_list{1});
dist=str2num(arg_list{2});
Y=rand(N, 1) > dist;
Y=[~Y Y];

W0 = str2num(arg_list{3});
W1 = str2num(arg_list{4});
W=[W0 W1];
J_history = costfunc(Y, W);;
WS_hist=W;
grads=gradientdesc(Y, W);

%Learning Schedule Parameters
LS=arg_list{5};
if(LS == "RM")
	fprintf('Selecting Robbins Monro Schedule\n');
	tao = str2num(arg_list{6})
	k = str2num(arg_list{7})
else
	fprintf('Selecting Ada Grad Schedule\n');
	eeta = str2num(arg_list{6})
	eps = str2num(arg_list{7})
endif
i = 1;
do
  oldcost = costfunc(Y, W);
  
  if(LS == "RM")
    alpha = (i+tao).^(k); %Monro Learning Schedule
  else
    %AdaGrad Learning
    ada_r = eeta./sqrt((eps)+sum(grads'.^2, 2));
  endif
  currgrad=gradientdesc(Y, W);
  grads = [grads; currgrad];

  if(LS == "RM")
    W = W-alpha*currgrad;
  else
    W = W-(ada_r'.*currgrad);
  endif
  curr_cost = costfunc(Y,W);
  J_history=[J_history; curr_cost];
  WS_hist=[WS_hist; W];
  i = i+1;
until(oldcost-curr_cost < 0.000001 || curr_cost == 0)

fprintf('Minimum value of the function is:');
costfunc(Y, W)
fprintf('Minimum value is at function:');
W
J_history;
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');
