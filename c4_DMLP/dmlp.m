clc,clear,close all
% Multi Layer Perceptrone
% sk.boo

% training set (iris dataset)
load iris.dat
x = iris(:,1:4);
y = iris(:,5)';
x = (x-mean(x))./std(x);

X = [ones(1,length(x)) ; x']; % bias add
one_hot = diag(ones(1,max(y)));
Y = one_hot(y,:)';

%number of nodes
in_node_n = length(x(1,:));
hd_node_n = 4;
hd2_node_n = 4;
out_node_n = length(Y(:,1));
error_epo = zeros(1,10);

%learning rate
lr = 0.2;

%weight matrix
U1 = rand(hd_node_n,in_node_n+1);
U2 = rand(hd2_node_n,hd_node_n+1);
U3 = rand(out_node_n,hd2_node_n+1);

o = zeros(size(Y));
epo = 0;
while 1
    sample_n = randi([length(X)*0.5,length(X)]);
    sample_index = randsample(length(X),sample_n);
    X_sample = X(:,sample_index);
    Y_sample = Y(:,sample_index);
    
    dU1 = zeros(size(U1));
    dU2 = zeros(size(U2));
    dU3 = zeros(size(U3));
    
    o = zeros(size(Y_sample));
    epo = epo+1;
    for i=1:sample_n
        %forward
        hidden_node = [1; Forward(@Sigmoid,X_sample(:,i),U1)];
        hidden2_node = [1; Forward(@Sigmoid,hidden_node,U2)];
        o(:,i) = Forward(@Sigmoid,hidden2_node,U3);
        
        %error backpropagation
        gradient1 = (Y_sample(:,i) - o(:,i)).*Forward(@d_Sigmoid,hidden2_node,U3);
        dU3 = dU3 -gradient1*hidden2_node';
        gradient2 = (gradient1'*U3(:,2:end))'.*Forward(@d_Sigmoid,hidden_node,U2);
        dU2 = dU2 -gradient2*hidden_node';
        gradient3 = (gradient2'*U2(:,2:end))'.*Forward(@d_Sigmoid,X_sample(:,i),U1);
        dU1 = dU1 -gradient3*X(:,i)';
    end
    
    %update weight
    U3 = U3 -lr*dU3/sample_n;
    U2 = U2 -lr*dU2/sample_n;
    U1 = U1 -lr*dU1/sample_n;
    
    error_epo(epo) = Mse(Y(:,sample_index),o,sample_n);
    fprintf("세대 : %6.0f    오차 : %5.4d\n",epo,error_epo(epo))
    %stop condition
    if error_epo(epo) < 1e-1 && epo >= 1e+4
        break
    end
end

figure()
plot((1:epo),error_epo)

%% test
for i=1:length(x)
    %forward
    hidden_node = [1; Forward(@Sigmoid,X(:,i),U1)];
    hidden2_node = [1; Forward(@Sigmoid,hidden_node,U2)];
    o(:,i) = Forward(@Sigmoid,hidden2_node,U3);
end
[~,max_o] = max(o);
error_epo = Mse(Y,o,length(x));
fprintf("출력 벡터 : %3.0f\n",max_o)
fprintf("\n오차 : %5.4d\n세대 : %6.0f\n",error_epo,epo)

%% function
%forward
function z = Forward(act,x,U)
z = act(U*x);
end

%objective function
function error = Mse(y,o,n)
error = sum(sum((y-o).^2))/n*2;
end
function error = Cee(y,o,n)
error = -sum(sum((y.*log(o)+(1-y).*log(1-o))))/n;
end

%activation function
function y = Sigmoid(s)
y = 1./(1+exp(-s));
end
function y = d_Sigmoid(s)
y = Sigmoid(s).*(1-Sigmoid(s));
end
function y = Relu(s)
y = max(s*0.01,s);
end
function y = d_Relu(s)
if s>0
    y=1;
else
    y=0.01;
end
end
