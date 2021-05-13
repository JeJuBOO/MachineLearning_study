clc,clear,close all
% Multi Layer Perceptrone
% sk.boo

% training set (iris dataset)
load iris.dat
x = iris([1:40, 51:90, 101:140],1:4);
y = iris([1:40, 51:90, 101:140],5)';
x = (x-mean(x))./std(x);

X = [ones(1,length(x)) ; x']; % bias add
one_hot = diag(ones(1,max(y)));
Y = one_hot(y,:)';

% n=1;
% % training set (iris dataset)
% figure(1)
% for i=1:4
%     for j=1:4
%         subplot(4,4,n)
%         plot(x(1:40,i),x(1:40,j),"r.",x(41:80,i),x(41:80,j),"g.",x(81:120,i),x(81:120,j),"k.")
%         n=n+1;
%     end
% end

%number of nodes
in_node_n = length(x(1,:));
hd_node_n = 4;
hd2_node_n = 3;
out_node_n = length(Y(:,1));
error_epo = zeros(1,10);
error = zeros(1,10);

%learning rate
lr = 0.2;

%weight matrix
U1 = rand(hd_node_n,in_node_n+1);
U2 = rand(hd2_node_n,hd_node_n+1);
U3 = rand(out_node_n,hd2_node_n+1);

epo = 0;
tic
sample_n = 10;
succese_n = 0;
while 1
    index = 1:length(X);
    epo = epo+1;
    o = zeros(size(Y));
    for j=1:length(X)/sample_n
        sample_index = randsample(length(index),sample_n);
        sample = index(sample_index);
        index(sample_index) = [];
        
        X_sample = X(:,sample);
        Y_sample = Y(:,sample);
        
        dU1 = zeros(size(U1));
        dU2 = zeros(size(U2));
        dU3 = zeros(size(U3));
               
        for i=1:sample_n
            %forward
            hidden_node = [1; Forward(@Sigmoid,X_sample(:,i),U1)];
            hidden2_node = [1; Forward(@Sigmoid,hidden_node,U2)];
            o(:,sample(i)) = Forward(@Sigmoid,hidden2_node,U3);
            
            %error backpropagation
            gradient1 = (Y_sample(:,i) - o(:,sample(i))).*Forward(@d_Sigmoid,hidden2_node,U3);
            dU3 = dU3 -gradient1*hidden2_node';
            gradient2 = (gradient1'*U3(:,2:end))'.*Forward(@d_Sigmoid,hidden_node,U2);
            dU2 = dU2 -gradient2*hidden_node';
            gradient3 = (gradient2'*U2(:,2:end))'.*Forward(@d_Sigmoid,X_sample(:,i),U1);
            dU1 = dU1 -gradient3*X_sample(:,i)';
        end
        
        %update weight
        U3 = U3 -lr*dU3/sample_n;
        U2 = U2 -lr*dU2/sample_n;
        U1 = U1 -lr*dU1/sample_n;
        
    end
    error_epo(epo) = Mse(Y,o,length(x));
    fprintf("세대 : %6.0f    오차 : %5.4d\n",epo,error_epo(epo))
    
    %stop condition
    [~,max_o] = max(o);
    succese_n = succese_n + prod(max_o==y);
    
    if succese_n==20 || epo == 10000
        break
    end 
end
toc
figure(2)
plot((1:epo),error_epo)
title("오차(MSE)")
xlabel("세대(epoch)")
ylabel("오차(error)")


%% test
x = iris([41:50, 91:100, 141:150],1:4);
y = iris([41:50, 91:100, 141:150],5)';
x = (x-mean(x))./std(x);

X = [ones(1,length(x)) ; x']; % bias add
one_hot = diag(ones(1,max(y)));
Y = one_hot(y,:)';
o = zeros(size(Y));

for i=1:length(x)
    %forward
    hidden_node = [1; Forward(@Sigmoid,X(:,i),U1)];
    hidden2_node = [1; Forward(@Sigmoid,hidden_node,U2)];
    o(:,i) = Forward(@Sigmoid,hidden2_node,U3);
end

[~,max_o] = max(o);
error_epo = Mse(Y,o,length(x));
fail = 0;
for i=1:length(x)
    if max_o(i) ~= y(i)
        fail = fail + 1;
    end
end

fprintf("\n오차 : %5.4d\n세대 : %6.0f\n",error_epo,epo)
fprintf("test 집합에서 틀린 개수 : %d",fail);

o_1 = find(max_o==1);
o_2 = find(max_o==2);
o_3 = find(max_o==3);
n=1;
figure(3)
for i=1:4
    for j=1:4
        subplot(4,4,n)
        plot(x(o_1,i),x(o_1,j),"r."); hold on
        plot(x(o_2,i),x(o_2,j),"g."); hold on
        plot(x(o_3,i),x(o_3,j),"k."); hold on
        n=n+1;
    end
end

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
