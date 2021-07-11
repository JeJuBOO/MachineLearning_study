clc,clear,close all
% CNN / MNIST
% Revision date: 2021.07.02
% mnist\trainingData.mat
% mnist\trainingData.mat
% images : 28 x 28 x 60000
% labels : 1 x 60000
% sk.boo

%% set up
% training set (MNIST)
load mnist\trainingData.mat;

images = images(:,:,1:6000);
labels = labels(1:6000);

x = reshape(images, 28,28,1,[]);
one_hot = diag(ones(1,max(labels)+1));
y = one_hot(labels+1,:)';
% x(input images) : 28*28*1 60000
% y(label) : 10*60000
load mnist\testingData.mat;

imageDim = size(x,1);
labelClasses = size(y,1);
% 커널 크기
kernelSize1 = [3 3 1 10];
kernelSize2 = [4 4 10 10];


%풀링 사이즈
poolDim1 = 2;
poolDim2 = 2;

%모델
layerDim1 = ( imageDim - kernelSize1(1) + 1 )/poolDim1;
layerDim2 = ( layerDim1 - kernelSize2(1) + 1 )/poolDim2;
layerDim3 = layerDim2^2*kernelSize2(4);
kernelSize3 = [labelClasses layerDim3];

% 학습률
lr = 0.01;

%세대 수 (epoch)
epo = 3;

%배치 수
batch = 150;

% 커널
U1 = 1e-1*randn(kernelSize1(1),kernelSize1(2),kernelSize1(3),kernelSize1(4)); % a*b
U2 = 1e-1*randn(kernelSize2(1),kernelSize2(2),kernelSize2(3),kernelSize2(4)); % c*d
U3 = 1e-1*randn(kernelSize3(1),kernelSize3(2));  % r*q
B1 = zeros(kernelSize1(4), 1);
B2 = zeros(kernelSize2(4), 1);
B3 = zeros(labelClasses, 1);

tic
%%
for e = 1:epo
    % data shuffle
    p = randperm(length(x));
    
    idx=0;
    for i = 1:batch:(length(x)-batch+1)
        %Correlation(in,kernel,stride,padding)
        %Pooling(in,num,stride)
        X = x(:,:,:,p(i:i+batch-1)); %i*j
        Y = y(:,p(i:i+batch-1));
        X_num = size(X,4);
        idx=idx+1;
        
        dU1 = zeros(size(U1));% a*b
        dU2 = zeros(size(U2));% c*d
        dU3 = zeros(size(U3));% r*q
        dB1 = zeros(size(B1));
        dB2 = zeros(size(B2));
        dB3 = zeros(size(B3));
        
        % 입력데이터 정규화
        X = Normalization3(X);
        
        z1 = Correlation(X,U1,B1);
        z1 = Normalization3(z1);
        layer1 = Relu(z1);% m*n
        [pool_layer1,pool_idx1] = Pooling(layer1,poolDim1); % m'*n'
        
        z2 = Correlation(pool_layer1,U2,B2);
        z2 = Normalization3(z2);
        layer2 = Relu(z2);% o*p
        [pool_layer2,pool_idx2] = Pooling(layer2,poolDim2); % o'*p'
        
        flat_layer3 = reshape(pool_layer2,[],X_num); % q*1
        out_layer = U3*flat_layer3 + B3; % r*1
        out_layer = (out_layer-mean(out_layer))./std(out_layer);
        
        % softmax error
        out = exp(out_layer)./sum(exp(out_layer));
        error(idx,:) = sum(-sum(Y.*log(out)))/batch;
        
        %% back prop
        gradient3 = Y - out; %out error gradient
        
        % o*p / CONV - FOOL - FC
        gradient_FC2 = reshape(U3' * gradient3,layerDim2,layerDim2,kernelSize2(4),X_num);
        gradient2 = UpSampling(gradient_FC2,poolDim2,pool_idx2).*ReluGradient(z2);
        
        % m*n / CONV - FOOL - CONV
        gradient_CONV1 = Convolution(gradient2,U2);
        gradient1 = UpSampling(gradient_CONV1,poolDim1,pool_idx1).*ReluGradient(z1);
        
        dU3 = gradient3*flat_layer3';
        dB3 = sum(gradient3,2);
        [dU2,dB2] = Update_grad(dU2,dB2,pool_layer1,gradient2);
        [dU1,dB1] = Update_grad(dU1,dB1,X,gradient1);
        
        U3 = U3 - lr*dU3/batch;
        B3 = B3 - lr*dB3/batch;
        U2 = U2 - lr*dU2/batch;
        B2 = B2 - lr*dB2/batch;
        U1 = U1 - lr*dU1/batch;
        B1 = B1 - lr*dB1/batch;
        
        tex1 = mean(error);
        fprintf("전체 학습 오차: %0.4f\n",round(tex1,4))
    end
    lr = lr/2;
    time = toc;

    testim = reshape(images, [28,28,1,10000]);
    one_hot = diag(ones(1,max(labels)+1));
    testlabel = one_hot(labels+1,:)';
    
    z1 = Correlation(testim,U1,B1);
    z1 = Normalization3(z1);
    layer1 = Relu(z1);% m*n
    [pool_layer1,pool_idx1] = Pooling(layer1,poolDim1); % m'*n'
    
    z2 = Correlation(pool_layer1,U2,B2);
    z2 = Normalization3(z2);
    layer2 = Relu(z2);% o*p
    pool_layer2 = Pooling(layer2,poolDim2); % o'*p'
    
    flat_layer3 = reshape(pool_layer2,[],length(testim)); % q*1
    out_layer = U3*flat_layer3 + B3; % r*1
    out_layer = (out_layer-mean(out_layer))./std(out_layer);
    
    % softmax error
    
    out = exp(out_layer)./sum(exp(out_layer));
%     out = sum(out, 2)/length(testlabel);
    [~,preds] = max(out,[],1);
    
    acc = sum(preds==labels)/length(preds);
    fprintf('Accuracy is %f\n',acc);
    time
    plot(error);
  
end
toc
load mnist\testingData.mat;
X = reshape(images, [28,28,1,10000]);
one_hot = diag(ones(1,max(labels)+1));
Y = one_hot(labels+1,:)';

for i=1:length(X)
    %forward
    X(:,:,1,i) = Normalization3(X(:,:,1,i));
    z1 = Correlation(X(:,:,1,i),U1,B1);
    z1 = Normalization3(z1);
    layer1 = Relu(z1);% m*n
    pool_layer1 = Pooling(layer1,poolDim1,poolDim1); % m'*n'
    
    z2 = Correlation(pool_layer1,U2,B2);
    z2 = Normalization3(z2);
    layer2 = Relu(z2);% o*p
    pool_layer2 = Pooling(layer2,poolDim2,poolDim2); % o'*p'
    
    flat_layer3 = reshape(pool_layer2,[],1); % q*1
    out_layer = U3*flat_layer3; % r*1
    out_layer = (out_layer-mean(out_layer))./std(out_layer);
    out = exp(out_layer)/sum(exp(out_layer));
    results(i) = min( Y(:,i) == (out == max(out)));
end
error_epo = round(mean(results)*100,2);
fprintf("\n정확도 : %5.4f\n",error_epo)






