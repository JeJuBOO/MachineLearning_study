clc,clear,close all
% CNN / MNIST
% Revision date: 2021.07.01
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

% 커널 크기
U1_ab = [3 3 1 1];
U2_cd = [4 4 1 1];
U3_rq = [size(y,1) 25];

%풀링 사이즈
p1 = 2;
p2 = 2;

% 입력데이터 정규화
x(isnan(Normalization3(x))) = 0;
% 학습률
lr = 0.1;

%세대 수 (epoch)
epo = 5000;

%배치 수 
batch = 100;



% 커널
U1 = randn(U1_ab(1),U1_ab(2),U1_ab(3),U1_ab(4)); % a*b
U2 = randn(U2_cd(1),U2_cd(2),U2_cd(3),U2_cd(4)); % c*d
U3 = randn(U3_rq(1),U3_rq(2));  % r*q

%%
for z = 1:epo
    % data shuffle
    p = randperm(length(x));
    X = x(:,:,1,p(1:batch)); %i*j
    Y = y(:,p(1:batch));
    
    dU1 = zeros(size(U1));% a*b
    dU2 = zeros(size(U2));% c*d
    dU3 = zeros(size(U3));% r*q
    
    for i = 1 : batch
        %Forward(in,kernel,stride,padding)
        %Pooling(in,num,stride)
        X(:,:,1,i) = Normalization3(X(:,:,1,i));
        z1 = Correlation(X(:,:,1,i),U1);
        z1 = Normalization3(z1);
        layer1 = Relu(z1);% m*n
        pol_layer1 = Pooling(layer1,p1,p1); % m'*n'
        
        z2 = Correlation(pol_layer1,U2);
        z2 = Normalization3(z2);
        layer2 = Relu(z2);% o*p
        pol_layer2 = Pooling(layer2,p2,p2); % o'*p'
       
        flat_layer3 = reshape(pol_layer2,[],1); % q*1
        out_layer = FC(flat_layer3,U3); % r*1
        out_layer = (out_layer-mean(out_layer))./std(out_layer);
        out = Softmax(out_layer);
        error(i,:) = -sum(Y(:,i).*log(out));
        %% back prop
        gradient3 = Y(:,i) - out;
        %출력층 gradient3
        %FC층 U3
        gradient2 = Gradient(layer2,z2,gradient3,U3,p2);% o*p
        gradient1 = Gradient(layer1,z1,gradient2,U2,p1);% m*n
        
        dU3 = dU3 - gradient3*flat_layer3';
        dU2 = dU2 - Correlation(pol_layer1,gradient2);
        dU1 = dU1 - Correlation(X(:,:,1,i),gradient1);
    end
    U3 = U3 - lr*dU3/batch;
    U2 = U2 - lr*dU2/batch;
    U1 = U1 - lr*dU1/batch;
    
   tex1 = mean(error);
   fprintf("전체 학습 오차: %0.5f\n",round(tex1,4))
end

load mnist\testingData.mat;
X = reshape(images, [28,28,1,10000]);
one_hot = diag(ones(1,max(labels)+1));
Y = one_hot(labels+1,:)';

for i=1:length(X)
    %forward
        X(:,:,1,i) = Normalization3(X(:,:,1,i));
        z1 = Correlation(X(:,:,1,i),U1);
        z1 = Normalization3(z1);
        layer1 = Relu(z1);% m*n
        pol_layer1 = Pooling(layer1,p1,p1); % m'*n'
        
        z2 = Correlation(pol_layer1,U2);
        z2 = Normalization3(z2);
        layer2 = Relu(z2);% o*p
        pol_layer2 = Pooling(layer2,p2,p2); % o'*p'
       
        flat_layer3 = reshape(pol_layer2,[],1); % q*1
        out_layer = FC(flat_layer3,U3); % r*1
        out_layer = (out_layer-mean(out_layer))./std(out_layer);
        out = Softmax(out_layer);
        results(i) = min( Y(:,i) == (out(:,i) == max(out)));
end
error_epo = round(mean(results)*100,2);
fprintf("\n정확도 : %5.4f\n",error_epo)






