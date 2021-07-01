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
X = [ones(1,length(X)) ; X]; % bias add
X = (X-mean(X))./std(X);

%number of nodes
in_node_n = length(X(:,1));
hd_node_n = 15;
hd2_node_n = 10;
hd3_node_n = 10;
out_node_n = length(Y(:,1));

%learning rate
lr = 0.02;
alpha_m = 1; 
alpha_lr = 0.9999;
ep = 2;

%weight matrix
U1 = randn(hd_node_n,in_node_n);
U2 = randn(hd2_node_n,hd_node_n+1);
U3 = randn(hd3_node_n,hd2_node_n+1);
U4 = randn(out_node_n,hd3_node_n+1);

%momentum
v1 = 0;v2 = 0;v3 = 0;v4 = 0;
r1 = 0;r2 = 0;r3 = 0;r4 = 0;

t = 1;
tic
epo = 5000;
sample_n = 60;
for j=1:epo    
    sample_index = randperm(length(X));
    X_sample = X(:,sample_index(1:sample_n));
    Y_sample = Y(:,sample_index(1:sample_n));
    
    dU1 = zeros(size(U1));
    dU2 = zeros(size(U2));
    dU3 = zeros(size(U3));
    dU4 = zeros(size(U4));
    
    for i=1:sample_n
        %% forward
        hidden_node = [1; Forward_mlp(@Relu,X_sample(:,i),U1)];
%         hidden_node = (hidden_node-mean(hidden_node))./std(hidden_node);
        hidden2_node = [1; Forward_mlp(@Relu,hidden_node,U2)];
%         hidden2_node = (hidden2_node-mean(hidden2_node))./std(hidden2_node);
        hidden3_node = [1; Forward_mlp(@Relu,hidden2_node,U3)];
%         hidden3_node = (hidden3_node-mean(hidden3_node))./std(hidden3_node);
        o = Forward_mlp(@exp,hidden3_node,U4)/sum(Forward_mlp(@exp,hidden3_node,U4));
        %% Error
%         o_num(i) = find(o==max(o));
%         y_num(i) = find(Y_sample(:,i)==1);
        error(i,:) = -sum(Y_sample(:,i).*log(o));
        
        %% Momentum
        hU4 = U4 + alpha_m*v4;
        hU3 = U3 + alpha_m*v3;
        hU2 = U2 + alpha_m*v2;
        hU1 = U1 + alpha_m*v1;
        
        %% error backpropagation
        gradient1 = (Y_sample(:,i) - o);
        dU4 = dU4 -gradient1*hidden3_node';
        gradient2 = (gradient1'*hU4(:,2:end))'.*Forward_mlp(@ReluGradient,hidden2_node,hU3);
        dU3 = dU3 -gradient2*hidden2_node';
        gradient3 = (gradient2'*hU3(:,2:end))'.*Forward_mlp(@ReluGradient,hidden_node,hU2);
        dU2 = dU2 -gradient3*hidden_node';
        gradient4 = (gradient3'*hU2(:,2:end))'.*Forward_mlp(@ReluGradient,X_sample(:,i),hU1);
        dU1 = dU1 -gradient4*X_sample(:,i)';
    end
    g4 = dU4/sample_n;
    g3 = dU3/sample_n;
    g2 = dU2/sample_n;
    g1 = dU1/sample_n;
    
    %% Momentum
    v4 = alpha_m*v4 - (1-alpha_m)*g4; v4 = v4/(1-(alpha_m)^t);
    v3 = alpha_m*v3 - (1-alpha_m)*g3; v3 = v3/(1-(alpha_m)^t);
    v2 = alpha_m*v2 - (1-alpha_m)*g2; v2 = v2/(1-(alpha_m)^t);
    v1 = alpha_m*v1 - (1-alpha_m)*g1; v1 = v1/(1-(alpha_m)^t);
    %% RMSProp
    r4 = alpha_lr*r4 + (1-alpha_lr)*g4.*g4; r4 = r4/(1-(alpha_lr)^t);
    r3 = alpha_lr*r3 + (1-alpha_lr)*g3.*g3; r3 = r3/(1-(alpha_lr)^t);
    r2 = alpha_lr*r2 + (1-alpha_lr)*g2.*g2; r2 = r2/(1-(alpha_lr)^t);
    r1 = alpha_lr*r1 + (1-alpha_lr)*g1.*g1; r1 = r1/(1-(alpha_lr)^t);
    
    %% update weight
    U4 = U4 - (lr./((1e-10)+sqrt(r4))).*v4 - lr*ep*sign(U4);
    U3 = U3 - (lr./((1e-10)+sqrt(r3))).*v3 - lr*ep*sign(U3);
    U2 = U2 - (lr./((1e-10)+sqrt(r2))).*v2 - lr*ep*sign(U2);
    U1 = U1 - (lr./((1e-10)+sqrt(r1))).*v1 - lr*ep*sign(U1);
    
    clc
%     tex2 = mean(o_num == y_num);
    tex1 = mean(error);
    mse(j,1) = tex1;
    fprintf("학습 횟수 : %d번\n",j)
    fprintf("학습된 글자 수 : %d 개 (한 번 반복에 64개씩 학습을 진행합니다.)\n",j*sample_n)
%     fprintf("학습 데이터 손글씨 인식률 : %0.2f%%\n",round(tex2*100,4))
    fprintf("전체 학습 오차(MSE) : %0.5f\n",round(tex1,4))
    cla
    
    plot(mse);
    axis([0 inf 0 5])
    title("MSE")
    drawnow;
end

toc


%% test
load mnist\testingData.mat;
X = reshape(images,28^2,[]);% input : 28^2 x 10000
one_hot = diag(ones(1,max(labels)+1));
Y = one_hot(labels+1,:)';% output : 10 x 10000
X = [ones(1,length(X)) ; X]; % bias add
X = (X-mean(X))./std(X);

o = zeros(size(Y));

for i=1:length(X)
    %forward
    hidden_node = [1; Forward_mlp(@Relu,X(:,i),U1)];
    hidden2_node = [1; Forward_mlp(@Relu,hidden_node,U2)];
    hidden3_node = [1; Forward_mlp(@Relu,hidden2_node,U3)];
    o(:,i) =Forward_mlp(@exp,hidden3_node,U4)/sum(Forward_mlp(@exp,hidden3_node,U4));
     results(i) = min( Y(:,i) == (o(:,i) == max(o(:,i))));
end

[~,max_o] = max(o);
error_epo = round(mean(results)*100,2);
fail = 0;
for i=1:length(X)
    if max_o(i) ~= labels(i)
        fail = fail + 1;
    end
end

fprintf("\n정확도 : %5.4f\n",error_epo)
fprintf("test 집합에서 틀린 개수 : %d\n",fail);

%% function
%forward
function z = Forward_mlp(act,x,U)
z = act(U*x);
end



