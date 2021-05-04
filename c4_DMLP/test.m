clc,clear,close all
% Deep Multi Layer Perceptrone
% sk.boo

% training set (iris dataset)
load iris.dat
x = iris(:,1:4)';
y = iris(:,5)';
X = [ones(1,length(x)) ; x]; % bias add
one_hot = diag(ones(1,max(y)));
Y = one_hot(y,:)';


%number of nodes
in_node_n = length(x(:,1));
hd_node_n1 = 4;
hd_node_n2 = 3;
out_node_n = length(Y(:,1));
Z = [in_node_n,hd_node_n1,hd_node_n2,out_node_n];
Z_index = [0,in_node_n+1,in_node_n+hd_node_n1+2,in_node_n+hd_node_n1+hd_node_n2+3 ...
    ,in_node_n+hd_node_n1+hd_node_n2+out_node_n+4];
%learning rate
lr = 0.02;

%weight matrix
U1 = rand(1,hd_node_n1*(in_node_n+1));
U2 = rand(1,hd_node_n2*(hd_node_n1+1));
U3 = rand(1,out_node_n*(hd_node_n2+1));
U = [U1,U2,U3];

z = zeros(1,Z_index(end));
epo = 0;
while 1
    sample_n = randi([length(X)*0.5,length(X)]);
    sample_index = randsample(length(X),sample_n);
    X_sample = X(:,sample_index);
    Y_sample = Y(:,sample_index);
    
    dU1 = zeros(size(U1));
    dU2 = zeros(size(U2));
    
    o = zeros(size(Y_sample));
    epo = epo+1;
    
    for i=1:sample_n
        z(1:Z_index(2)) = X_sample(:,i);
        %forward
        for j=1:length(z)-1
          z(Z_index(j+1)+1:Z_index(j+2),1) = [1; ...
              forward(@sigmoid,z(Z_index(j)+1:Z_index(j+1)),reshape(U(),[Z(i+1),(Z(i)+1)]))];
        end
        %error backpropagation
        gradient2 = (Y_sample(:,i) - o(:,i)).*d_sigmoid(U2*hidden_node);
        dU2 = dU2 -gradient2*hidden_node';
        gradient1 = (gradient2'*U2(:,2:end))'.*d_sigmoid(U1*X_sample(:,i));
        dU1 = dU1 -gradient1*X_sample(:,i)';
    end
    %update weight
    U2 = U2 -lr*dU2/sample_n;
    U1 = U1 -lr*dU1/sample_n;
    
    %stop condition
    if mse(Y_sample,o) < 1e-2 & epo >= 1e+5
        break
    end
end

for i=1:length(X)
    %forward
    hidden_node = [1; sigmoid(U1*X(:,i))];
    o(:,i) = sigmoid(U2*hidden_node);
    
end

fprintf("출력 벡터 : %5.4f %5.4f %5.4f %5.4f\n",o)
fprintf("오차 : %5.4d\n세대 : %6.0f\n",mse(Y,o),epo)
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







