clc,clear,close all
% CNN / MNIST
% Revision date: 2021.07.02
% mnist\trainingData.mat
% images : 28 x 28 x 60000
% labels : 1 x 60000
% mnist\testingData.mat
% images : 28 x 28 x 6000
% labels : 1 x 6000
% sk.boo

%% set up
% training set (MNIST)
load mnist\trainingData.mat;
% 학습데이터 간소화
% images = images(:,:,1:6000);
% labels = labels(1:6000);
x = reshape(images, 28,28,1,[]);
one_hot = diag(ones(1,max(labels)+1));
y = one_hot(labels+1,:)';

% test set (MNIST)
load mnist\testingData.mat;

% 커널 크기
kernelSize1 = [3 3 1 10];
kernelSize2 = [4 4 10 10];

%풀링 사이즈
poolDim1 = 2;
poolDim2 = 2;

%모델
imageDim = size(x,1);
labelClasses = size(y,1);
layerDim1 = ( imageDim - kernelSize1(1) + 1 )/poolDim1;
layerDim2 = ( layerDim1 - kernelSize2(1) + 1 )/poolDim2;
layerDim3 = layerDim2^2*kernelSize2(4);
kernelSize3 = [labelClasses layerDim3];

% 학습률
lr = 0.01;
% 가중치 감쇠
lambda = 0.0001;
% RMSProp
alpha_r = 0.9;

%세대 수 (epoch)
epo = 5;

%배치 수
batch = 150;

% 커널
U1 = 1e-1*randn(kernelSize1(1),kernelSize1(2),kernelSize1(3),kernelSize1(4)); % a*b
U2 = 1e-1*randn(kernelSize2(1),kernelSize2(2),kernelSize2(3),kernelSize2(4)); % c*d
U3 = 1e-1*randn(kernelSize3(1),kernelSize3(2));  % r*q
B1 = zeros(kernelSize1(4), 1);
B2 = zeros(kernelSize2(4), 1);
B3 = zeros(labelClasses, 1);
%momentum
r1 = zeros(size(U1));% a*b
r2 = zeros(size(U2));% c*d
r3 = zeros(size(U3));% r*q
rb1 = zeros(size(B1));
rb2 = zeros(size(B2));
rb3 = zeros(size(B3));

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
%         X =  Relu(Normalization3(X));
        X = X/255;
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
        wCost = lambda/2 * (sum(U3(:).^2)+sum(U2(:).^2)+sum(U1(:).^2));
        error(idx*e,:) = sum(-sum(Y.*log(out)))/batch + wCost;

        %% Backpropagation
        gradient3 = out - Y; %out error gradient
        
        % o*p / CONV - FOOL - FC
        gradient_FC2 = reshape(U3' * gradient3,layerDim2,layerDim2,kernelSize2(4),X_num);
        gradient2 = UpSampling(gradient_FC2,poolDim2,pool_idx2).*ReluGradient(z2);
        
        % m*n / CONV - FOOL - CONV
        gradient_CONV1 = Convolution(gradient2,U2);
        gradient1 = UpSampling(gradient_CONV1,poolDim1,pool_idx1).*ReluGradient(z1);
        
        % 누적 그레디언트
        dU3 = gradient3*flat_layer3';
        dB3 = sum(gradient3,2);
        [dU2,dB2] = Update_grad(dU2,dB2,pool_layer1,gradient2);
        [dU1,dB1] = Update_grad(dU1,dB1,X,gradient1);

         
        %% RMSProp
        r3 =  alpha_r*r3 +(1-alpha_r)*(dU3/batch+lambda*U3).^2;
        rb3 =  alpha_r*rb3 +(1-alpha_r)*(dB3/batch).^2;
        r2 =  alpha_r*r2 +(1-alpha_r)*(dU2/batch+lambda*U2).^2;
        rb2 =  alpha_r*rb2 +(1-alpha_r)*(dB2/batch).^2;
        r1 =  alpha_r*r1 +(1-alpha_r)*(dU1/batch+lambda*U1).^2;
        rb1 =  alpha_r*rb1 + (1- alpha_r)*(dB1/batch).^2;
              
        U3 = U3 - (lr./(1e-4+sqrt(r3))).*(dU3/batch+lambda*U3);
        B3 = B3 - (lr./(1e-4+sqrt(rb3))).*dB3/batch;
        U2 = U2 - (lr./(1e-4+sqrt(r2))).*(dU2/batch+lambda*U2);
        B2 = B2 - (lr./(1e-4+sqrt(rb2))).*dB2/batch;
        U1 = U1 - (lr./(1e-4+sqrt(r1))).*(dU1/batch+lambda*U1);
        B1 = B1 - (lr./(1e-4+sqrt(rb1))).*dB1/batch;
        
        tex1 = mean(error);
        fprintf("%2.0f epoch / 진행도 : %2.4f %% 전체 학습 오차: %0.4f\n",e,i/length(x)*100,round(tex1,4))
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
    [~,preds] = max(out,[],1);
    
    acc = sum((preds-1)==labels)/length(preds);
    fprintf('%2.0f epoch / Accuracy is %4.2f %%\n',e,acc*100);
    time
    plot(error,'y');hold on
    plot(smoothdata(error),'r','LineWidth',1)
  
end






