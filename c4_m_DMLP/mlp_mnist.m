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

% test
load mnist\testingData.mat;
X_test = reshape(images,28^2,[]);% input : 28^2 x 10000
one_hot = diag(ones(1,max(labels)+1));
Y_test = labels;
X_test = X_test/255;

% load iris.dat
% X = iris([1:40, 51:90, 101:140],1:4)';
% Y = iris([1:40, 51:90, 101:140],5)';
% one_hot = diag(ones(1,max(Y)));
% Y = one_hot(Y,:)';% output : 10 x 60000
% X = (X-mean(X))./std(X);

%number of nodes
in_node_n = length(X(:,1));
hd_node_n = 15;
hd2_node_n = 20;
out_node_n = length(Y(:,1));

%learning rate
lr = 0.02;

%weight matrix
U1 = 0.1*randn(in_node_n,hd_node_n);
U2 = 0.1*randn(hd_node_n,hd2_node_n);
U3 = 0.1*randn(hd2_node_n,out_node_n);
B1 = zeros(hd_node_n,1);
B2 = zeros(hd2_node_n,1);
B3 = zeros(out_node_n,1);


tic
epo = 100;
batchSize = 100;
idx = 0;
for e = 1:epo
    % data shuffle
    p = randperm(length(X));
    
    for i = 1:batchSize:(length(X)-batchSize+1)
        X_sample = X(:,p(i:i+batchSize-1));
        Y_sample = Y(:,p(i:i+batchSize-1))';
        X_num = size(X_sample,4);
        idx=idx+1;
        
        %% forward
       
        hidden_node = Relu(X_sample'*U1+B1');
        hidden2_node = Relu(hidden_node*U2+B2');
        hidden3 = hidden2_node*U3+B3';
        o = exp(hidden3)./sum(exp(hidden3),2);
        %% Error
        error(idx,:) = mean(-sum(Y_sample.*log(o)));
        
        %% error backpropagation
        gradient3 = o - Y_sample;         
        gradient2 = gradient3*U3'.*ReluGradient(hidden2_node);
        gradient1 = gradient2*U2'.*ReluGradient(hidden_node);
        
        dU3 = (hidden2_node'*gradient3);
        dB3 = sum(gradient3,1);
        dU2 = (hidden_node'*gradient2);
        dB2 = sum(gradient2,1);
        dU1 = (X_sample*gradient1);
        dB1 = sum(gradient1,1);

        %% update weight

        U3 = U3 - lr*dU3/batchSize;
        B3 = B3 - lr*dB3'/batchSize;
        U2 = U2 - lr*dU2/batchSize;
        B2 = B2 - lr*dB2'/batchSize;
        U1 = U1 - lr*dU1/batchSize;
        B1 = B1 - lr*dB1'/batchSize;
    end
    
    %% test forward
    hidden_node = Relu(X_test'*U1+B1');
    hidden2_node = Relu(hidden_node*U2+B2');
    hidden3 = hidden2_node*U3+B3';
    o = exp(hidden3)./sum(exp(hidden3),2);
    [~,preds] = max(o,[],2);
    
    acc(e) = sum((preds'-1)==Y_test)/length(preds);
    fprintf('%2.0f epoch / Accuracy is %4.2f %%\n',e,acc(e)*100);
    
end
toc
figure(1)
plot(e/length(error):e/length(error):e,error,'c');hold on;
xlabel("Epoch");ylabel("Cost");
plot(e/length(error):e/length(error):e,smoothdata(error),'b','LineWidth',1.5)
hold off;
figure(2)
plot(1:e,acc,'r')
title('Accuracy');
xlabel("Epoch");ylabel("Accuracy");