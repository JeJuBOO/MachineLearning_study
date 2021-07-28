clc,clear,close all
% MLP / MNIST
% Revision date: 2021.05.19
% mnist\trainingData.mat
% images : 28 x 28 x 60000
% labels : 1 x 60000
% sk.boo

% training set (MNIST)
load mnist\trainingData.mat;
X = reshape(images,28^2,[]);% input : 28^2 x 60000
one_hot = diag(ones(1,max(labels)+1));
Y = one_hot(labels+1,:)';% output : 10 x 60000
X = X/255;

% load iris.dat
% X = iris([1:40, 51:90, 101:140],1:4)';
% Y = iris([1:40, 51:90, 101:140],5)';
% one_hot = diag(ones(1,max(Y)));
% Y = one_hot(Y,:)';% output : 10 x 60000
% X = (X-mean(X))./std(X);

%number of nodes
in_node_n = length(X(:,1));
hd_node_n = 15;
hd2_node_n = 12;
hd3_node_n = 10;
out_node_n = length(Y(:,1));

%learning rate
lr = 0.2;
alpha_m = 0;
alpha_lr = 0.9;
ep = 1;

%weight matrix
U1 = randn(hd_node_n,in_node_n);
U2 = randn(hd2_node_n,hd_node_n);
U3 = randn(hd3_node_n,hd2_node_n);
U4 = randn(out_node_n,hd3_node_n);
B1 = zeros(hd_node_n,1);
B2 = zeros(hd2_node_n,1);
B3 = zeros(hd3_node_n,1);
B4 = zeros(out_node_n,1);
%momentum
v1 = 0;v2 = 0;v3 = 0;v4 = 0;
r1 = 0;r2 = 0;r3 = 0;r4 = 0;


tic
epo = 5000;
batchSize = 100;
idx = 0;
for e = 1:epo
    % data shuffle
    p = randperm(length(X));
    
    for i = 1:batchSize:(length(X)-batchSize+1)
        X_sample = X(:,p(i:i+batchSize-1));
        Y_sample = Y(:,p(i:i+batchSize-1));
        X_num = size(X_sample,4);
        idx=idx+1;
        
        %% forward
       
        hidden_node = Relu(U1*X_sample+B1);
        hidden2_node = Relu(U2*hidden_node+B2);
        hidden3_node = Relu(U3*hidden2_node+B3);
        hidden4 = U4*hidden3_node+B4;
        
        o = exp(hidden4)./sum(exp(hidden4),1);
        %% Error
%         o_num(i) = find(o==max(o));
%         y_num(i) = find(Y_sample(:,i)==1);
        error(idx,:) = -sum(sum(Y_sample.*log(o)))/batchSize;
        
        %% error backpropagation
        gradient4 = (o - Y_sample);     
        gradient3 = (gradient4'*U4)'.*ReluGradient(U3*hidden2_node);       
        gradient2 = (gradient3'*U3)'.*ReluGradient(U2*hidden_node);
        gradient1 = (gradient2'*U2)'.*ReluGradient(U1*X_sample);
        
        dU4 = (gradient4*hidden3_node')/batchSize;
        dB4 = sum(gradient4,2)/batchSize;
        dU3 = (gradient3*hidden2_node')/batchSize;
        dB3 = sum(gradient3,2)/batchSize;
        dU2 = (gradient2*hidden_node')/batchSize;
        dB2 = sum(gradient2,2)/batchSize;
        dU1 = (gradient1*X_sample')/batchSize;
        dB1 = sum(gradient1,2)/batchSize;
        
%         %% Momentum
%         v4 = alpha_m*v4 + (1-alpha_m)*dU4; v4 = v4/(1-(alpha_m)^idx);
%         v3 = alpha_m*v3 + (1-alpha_m)*dU3; v3 = v3/(1-(alpha_m)^idx);
%         v2 = alpha_m*v2 + (1-alpha_m)*dU2; v2 = v2/(1-(alpha_m)^idx);
%         v1 = alpha_m*v1 + (1-alpha_m)*dU1; v1 = v1/(1-(alpha_m)^idx);
%         %% RMSProp
%         r4 = alpha_lr*r4 + (1-alpha_lr)*dU4.*dU4; r4 = r4/(1-(alpha_lr)^idx);
%         r3 = alpha_lr*r3 + (1-alpha_lr)*dU3.*dU3; r3 = r3/(1-(alpha_lr)^idx);
%         r2 = alpha_lr*r2 + (1-alpha_lr)*dU2.*dU2; r2 = r2/(1-(alpha_lr)^idx);
%         r1 = alpha_lr*r1 + (1-alpha_lr)*dU1.*dU1; r1 = r1/(1-(alpha_lr)^idx);
%         
%         %% update weight
%         U4 = U4 - (lr./((1e-10)+sqrt(r4))).*v4 - lr*ep*sign(U4);
%         U3 = U3 - (lr./((1e-10)+sqrt(r3))).*v3 - lr*ep*sign(U3);
%         U2 = U2 - (lr./((1e-10)+sqrt(r2))).*v2 - lr*ep*sign(U2);
%         U1 = U1 - (lr./((1e-10)+sqrt(r1))).*v1 - lr*ep*sign(U1);
        %% update weight
        U4 = U4 - lr*dU4 - lr*ep*sign(U4);
        B4 = B4 - lr*dB4;
        U3 = U3 - lr*dU3 - lr*ep*sign(U3);
        B3 = B3 - lr*dB3;
        U2 = U2 - lr*dU2 - lr*ep*sign(U2);
        B2 = B2 - lr*dB2;
        U1 = U1 - lr*dU1 - lr*ep*sign(U1);
        B1 = B1 - lr*dB1;
        
        fprintf("%2.0f epoch 진행도 : %2.4f %% 전체 학습 오차: %0.4f\n",e,i/length(X)*100,error(idx))
    end
%% test
load mnist\testingData.mat;
X_test = reshape(images,28^2,[]);% input : 28^2 x 10000
one_hot = diag(ones(1,max(labels)+1));
Y_test = one_hot(labels+1,:)';% output : 10 x 10000
X_test = X_test/255;
%% forward
hidden_node = Relu(U1*X_test+B1);
hidden2_node = Relu(U2*hidden_node+B2);
hidden3_node = Relu(U3*hidden2_node+B3);
hidden4 = U4*hidden3_node+B4;

o = exp(hidden4)./sum(exp(hidden4),1);
[~,preds] = max(o,[],1);

acc = sum((preds-1)==labels)/length(preds);
fprintf('%2.0f epoch / Accuracy is %4.2f %%\n',e,acc*100);
end

toc


%% test
load mnist\testingData.mat;
X = reshape(images,28^2,[]);% input : 28^2 x 10000
one_hot = diag(ones(1,max(labels)+1));
Y = one_hot(labels+1,:)';% output : 10 x 10000
X = [ones(1,length(X)) ; X]; % bias add
X = (X-mean(X))./std(X);

% X = iris([41:50, 91:100, 141:150],1:4)';
% labels = iris([41:50, 91:100, 141:150],5)';
% one_hot = diag(ones(1,max(labels)));
% Y = one_hot(labels,:)';% output : 10 x 60000
% X = (X-mean(X))./std(X);

o = zeros(size(Y));
fail = 0;
for i=1:length(X)
    %forward
    hidden_node = [1; Forward_mlp(@Relu,X(:,i),U1)];
    hidden2_node = [1; Forward_mlp(@Relu,hidden_node,U2)];
    hidden3_node = [1; Forward_mlp(@Relu,hidden2_node,U3)];
    o(:,i) =Forward_mlp(@exp,hidden3_node,U4)/sum(Forward_mlp(@exp,hidden3_node,U4));
    results(i) = min( Y(:,i) == (o(:,i) == max(o(:,i))));
    [~,max_o] = max(o(:,i));
    if max_o(i) ~= labels(i)
        fail = fail + 1;
    end

end

error_epo = round(mean(results)*100,2);

plot(mse);
axis([0 inf 0 5])
title("MSE")
fprintf("\n정확도 : %5.4f\n",error_epo)
fprintf("test 집합에서 틀린 개수 : %d\n",fail);

%% function
%forward
function z = Forward_mlp(act,x,U)
z = act(U*x);
end
function z = Normalization(x)
z = (x-mean(x))./std(x);
end

