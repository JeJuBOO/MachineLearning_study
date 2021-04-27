clc,clear,close all
% Multi Layer Perceptrone
% sk.boo

% training set (XOR)
x = [0,0 ; 1,0 ; 0,1 ; 1,1]';
Y = [0,1,1,0];
X = [ones(1,length(x)) ; x]; % bias add

%number of nodes
in_node_n = length(x(:,1));
hd_node_n = 4;
out_node_n = length(Y(:,1));

%learning rate
lr = 0.8;

%weight matrix
U1 = rand(hd_node_n,in_node_n+1);
U2 = rand(out_node_n,hd_node_n+1);

o = zeros(size(Y));
epo = 0;
while 1
    
    epo = epo+1;
    for i=1:length(x)
        %forward
        hidden_node = [1; sigmoid(U1*X(:,i))];
        o(:,i) = sigmoid(U2*hidden_node);
        
        %error backpropagation
        gradient2 = (Y(:,i) - o(:,i)).*d_sigmoid(U2*hidden_node);
        dU2 = -gradient2*hidden_node';
        gradient1 = (gradient2'*U2(:,2:end))'.*d_sigmoid(U1*X(:,i));
        dU1 = -gradient1*X(:,i)';
        
        %update weight
        U2 = U2 -lr*dU2;
        U1 = U1 -lr*dU1;
    end
    error(epo) = mse(Y,o);
    %stop condition
    if mse(Y,o) < 1e-4 & epo >= 1e+3
        break
    end
end

fprintf("출력 벡터 : %5.4f %5.4f %5.4f %5.4f\n",o)
fprintf("오차 : %5.4d\n세대 : %6.0f",mse(Y,o),epo)
plot((1:epo),error)

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




