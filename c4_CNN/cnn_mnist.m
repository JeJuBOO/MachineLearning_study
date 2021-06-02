clc,clear,close all
% CNN / MNIST
% Revision date: 2021.05.19
% mnist\trainingData.mat
% mnist\trainingData.mat
% images : 28 x 28 x 60000 
% labels : 1 x 60000
% sk.boo

%% set up
% training set (MNIST)
load mnist\trainingData.mat;
x = reshape(images, [28,28,1,60000]);
one_hot = diag(ones(1,max(labels)+1));
y = one_hot(labels+1,:)';
% x(input images) : 28*28*1 60000
% y(label) : 10*60000

% % 입력데이터 정규화
x = (x-mean(x,[1,2,3]))./std(x,0,[1,2,3]);
x(isnan(x)) = 0;
% 학습률
lr = 0.1;

%세대 수 (epoch)
epo = 4;

%배치 수 
batch = 1;

% 커널
kn_1 = randn(3,3,1,8);
kn_2 = randn(3,3,1,16);
kn_3 = randn(3,3,1,32);
u = randn(10,7*7*128*32);

%%
for z = 1:epo
    % data shuffle
    p = randperm(length(x));
    X = x(:,:,1,p(1:batch));
    Y = y(:,p(1:batch));
    
    dkn_1 = zeros(size(kn_1));
    dkn_2 = zeros(size(kn_2));
    dkn_3 = zeros(size(kn_3));
    du = zeros(size(u));
    for i = 1 : batch
        %Forward(in,kernel,act_func,stride,padding)
        %Pooling(in,num,stride)
        layer1 = Forward(X(:,:,1,i),kn_1,@Relu,1,1);
        pol_layer1 = Pooling(layer1,2,2);
        
        layer2 = Forward(pol_layer1,kn_2,@Relu,1,1);
        pol_layer2 = Pooling(layer2,2,2);
        
        layer3 = Forward(pol_layer2,kn_3,@Relu,1,1);

        out_layer = FC(layer3,u);
        out_layer = (out_layer-mean(out_layer))./std(out_layer);
        o = Softmax(out_layer);
        
        
        gradient1 = o-Y(:,i);
        gradient2 = reshape(gradient1'*u,size(layer3)).*ReluGradient(layer3);
        gradient3 = gradient2'*kn_3.*ReluGradient(layer2);
        du = du - gradient1*reshape(layer3,[],1)';
        
        
        dkn_3 = dkn_3 - gradient2*u';
        gradient3 = (gradient2'*U2(:,2:end))'.*Forward(@d_Sigmoid,X_sample(:,i),U1);
        dU1 = dU1 -gradient3*X_sample(:,i)';
        
        
    end
end























