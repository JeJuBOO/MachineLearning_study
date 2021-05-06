clc,clear,close all
% Multi Layer Perceptrone
% sk.boo

% training set (iris dataset)
% load iris.dat
% x = iris(:,1:4)';
% y = iris(:,5);
% X = [ones(1,length(x)) ; x]; % bias add
% one_hot = diag(ones(1,max(y)));
% Y = one_hot(y,:)';

% training set (XOR)
x = [0,0 ; 1,0 ; 0,1 ; 1,1]';
Y = [0,1,1,0];
X = [ones(1,length(x)) ; x]; % bias add

%number of nodes
in_node_n = length(x(:,1));
hd_node_n = 4;
out_node_n = length(Y(:,1));

%learning rate
lr = 0.02;

%weight matrix
U1 = rand(hd_node_n,in_node_n+1);
U2 = rand(out_node_n,hd_node_n+1);


epo = 0;
while 1
    sample_n = randi(length(X));
    sample_index = randsample(length(X),sample_n);
    X_sample = X(:,sample_index);
    Y_sample = Y(:,sample_index);
    
    dU1 = zeros(size(U1));
    dU2 = zeros(size(U2));
    o = zeros(size(Y_sample));
    epo = epo+1;
    for i=1:sample_n
        %forward
        hidden_node = [1; sigmoid(U1*X_sample(:,i))];
        o(:,i) = sigmoid(U2*hidden_node);
        
        %error backpropagation
        gradient2 = (Y_sample(:,i) - o(:,i)).*d_sigmoid(U2*hidden_node);
        dU2 = dU2 -gradient2*hidden_node';
        gradient1 = (gradient2'*U2(:,2:end))'.*d_sigmoid(U1*X_sample(:,i));
        dU1 = dU1 -gradient1*X_sample(:,i)';
    end
    %update weight
    U2 = U2 -lr*dU2/sample_n;
    U1 = U1 -lr*dU1/sample_n;
    
    error(epo) = mse(Y_sample,o);
    %stop condition
    if mse(Y_sample,o) < 1e-2 & epo >= 1e+5
        break
        break
    end
end

for i=1:length(X)
    %forward
    hidden_node = [1; sigmoid(U1*X(:,i))];
    o(:,i) = sigmoid(U2*hidden_node);
end

plot((1:epo),error)
fprintf("출력 벡터 : %5.4f %5.4f %5.4f %5.4f\n",o)
fprintf("오차 : %5.4d\n세대 : %6.0f",mse(Y,o),epo)
%objective function
function error = mse(y,o)
error = 0.5*sum(sum((y-o).^2));
end

%activation function
function y = sigmoid(s)
y = 1./(1+exp(-s));
end
function y = d_sigmoid(s)
y = sigmoid(s).*(1-sigmoid(s));
end




