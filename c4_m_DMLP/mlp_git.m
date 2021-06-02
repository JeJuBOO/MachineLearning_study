% *********************************************
% MNIST Neural Networks
% @author: Deep.I Inc. @Jongwon Kim
% deepi.contact.us@gmail.com
% Revision date: 2020-12-01
% See here for more information :
%    https://deep-eye.tistory.com
%    https://deep-i.net
% **********************************************
% STRUCTURE : BATCH MLP
% input : 28 x 28 x 60000 
% output : 1 x 60000
% MODE : Batch
% ACTIVATION FUNCTION : 'Relu'
% ERROR RATE : 2.51

%% F5를 눌러서 실행해주세요.

clear,clc,cla,close all
% input("\n\n 퍼셉트론을 활용한 손글씨 인식 프로그램 입니다. [엔터키를 눌러주세요] ")

%%
load mnist\trainingData.mat;

images = reshape(images,28^2,[]);
one_hot = diag(ones(1,max(labels)+1));
labels = one_hot(labels+1,:)';
% input : 28^2 x 60000 
% output : 10 x 60000
clc
% input("\n\n 배포해드린 숫자 데이터를 로드할게요. [엔터키를 눌러주세요] ")

%% 학습 데이터를 한번 봅시다
% mnist = images(:,1:200);
% mnist = reshape(mnist,28,28,200);
% montage(mnist)
% 
% title("학습에 활용할 손글씨 이미지 입니다. [엔터키를 눌러주세요] ")
% % clc
% input("\n\n 학습에 활용할 손글씨 이미지 입니다. [엔터키를 눌러주세요] ")

x = images;
x = (x-mean(x))./std(x);
y = labels';
cols = size(x,2);
imRe = 28;

% clc
% fprintf("데이터의 개수 : %d \n이미지 해상도 : %d x %d\n입력 차원 : %d\n",cols,imRe,imRe,imRe^2)
% alpha = input("\n\n 학습의 정도를 결정하는 Learning Rate을 입력해주세요. [0.1~ 0.0001] ");
% clc
% pNum_1  = input("\n\n 첫번째 층의 퍼셉트론(뉴런)의 개수를 입력해주세요. [1~inf] ");
% pNum_2  = input(" 두번째 층의 퍼셉트론(뉴런)의 개수를 입력해주세요. [1~inf] ");
% eh  = input(" 학습을 몇번 반복할지 반복 횟수를 입력해주세요. [1~inf] ");
% fprintf("\n\n %d-%d-%d-%d 의 구조를 갖는 신경망이 완성되었습니다.",imRe^2,pNum_1,pNum_2,10);
% input(" 엔터를 누루면 학습을 시작합니다!");
alpha = 0.2;
a1 = 0.9;
a2 = 0.9;
pNum_1 = 20;
pNum_2 = 15;
eh = 1000;
% fprintf("\n\n %d-%d-%d-%d 의 구조를 갖는 신경망이 완성되었습니다.",imRe^2,pNum_1,pNum_2,10);
% input(" 엔터를 누루면 학습을 시작합니다!");

% 1st Layer
node_1w = fc_node('weight', imRe^2, pNum_1);
node_1b = fc_node('bias', pNum_1,1);
% 2nd Layer
node_2w = fc_node('weight', pNum_1, pNum_2);
node_2b = fc_node('bias', pNum_2,1);
% 3rd Layer
node_3w = fc_node('weight', pNum_2, 10);
node_3b = fc_node('bias', 10,1);
% batch size
batch =64;
v1=0;v2=0;v3=0;
r1=0;r2=0;r3=0;
close all
for z = 1 : eh
    
    % data shuffle
    p = randperm(cols);
    X = x(:,p(1:batch));
    Y = y(p(1:batch),:);
    
    % batch memory init (weight)
    batch_1 = 0; batch_2 = 0; batch_3 = 0;
    % batch memory init (bias)
    batch_4 = 0; batch_5 = 0; batch_6 = 0;
    t = 1;
    for i = 1 : batch
        
        %% Feed Forward propagation
        
        f1 = Relu(X(:,i)' * node_1w + node_1b');
        f1 = (f1-mean(f1))./std(f1);
        f2 = Relu(f1 * node_2w + node_2b');
        f2 = (f2-mean(f2))./std(f2);
        f3 = exp(f2 * node_3w + node_3b') / sum(exp(f2 * node_3w + node_3b')) ;
        %% Error
        P(i) = find(f3==max(f3));
        O(i) = find(Y(i,:)==max(Y(i,:)));
        E(i,:) = - sum(Y(i,:).*log(f3));
        
        
       %% graidant
       
        b3 = f3 - Y(i,:);
        b2 = b3 * node_3w' .* ReluGradient(f2);
        b1 = b2 * node_2w' .* ReluGradient(f1);
       
        g3 = f2' * b3;
        g2 = f1' * b2;
        g1 = X(:,i) * b1;
        
        %모멘텀
        v1 = a1*v1-(1-a1)* g3; v1 = v1/(1-a1^t);
        
        v2 = a1*v2-(1-a1)* g2; v2 = v2/(1-a1^t);
        
        v3 = a1*v3-(1-a1)* g1; v3 = v3/(1-a1^t);
        
        %RMSProp
        r1 = a2*r1 + (1-a2)*g3.*g3;  r1 = r1/(1-a2^t);
        
        r2 = a2*r2 + (1-a2)*g2.*g2;  r2 = r2/(1-a2^t);
        
        r3 = a2*r3 + (1-a2)*g1.*g1;  r3 = r3/(1-a2^t);
        
        d_g1 = -alpha*v1/((1e-4)+sqrt(r1));
        d_g2 = -alpha*v2/((1e-4)+sqrt(r2));
        d_g3 = -alpha*v3/((1e-4)+sqrt(r3));
       %% Batch
        batch_1 = batch_1 + d_g1;
        batch_4 = batch_4 + (r1 * b3)';
        
        batch_2 = batch_2 + d_g2;
        batch_5 = batch_5 + (r2 * b2)';
        
        batch_3=  batch_3 + d_g3;
        batch_6 = batch_6 + (r3 * b1)' ;
        
    end
    
    %% Update
    node_3w = node_3w - batch_1 / batch;
    node_2w = node_2w - batch_2 / batch;
    node_1w = node_1w - batch_3 / batch;
    
    node_3b = node_3b - batch_4 / batch;
    node_2b = node_2b - batch_5 / batch;
    node_1b = node_1b - batch_6 / batch;
    
    t=t+1;
    %% 그래프 보기
    tex2 = mean(P == O);
    tex1 = mean(E);
    mse(z,1) = mean(E);
    format shortG
    clc
    fprintf("학습 횟수 : %d번\n",z)
    fprintf("학습된 글자 수 : %d 개 (한 번 반복에 64개씩 학습을 진행합니다.)\n",z*batch)
    fprintf("학습 데이터 손글씨 인식률 : %0.2f%%\n",round(tex2*100,4))
    fprintf("전체 학습 오차(MSE) : %0.5f",round(tex1,4))
    cla
    figure(1)
    plot(mse);
    axis([0 inf 0 5])
    title("MSE")
    drawnow;
%     subplot(1,2,2)
%     
%     testing = reshape(X,28,28,batch);
%     montage(testing(:,:,1:batch));
%     title("학습중인 숫자")
%     drawnow;
%     %
end
clc
fprintf("학습 횟수 : %d번\n",z)
fprintf("학습된 글자 수 : %d 개 (한 번 반복에 64개씩 학습을 진행합니다.)\n",z*batch)
fprintf("학습 데이터 손글씨 인식률 : %0.2f%%\n",round(tex2*100,4))
fprintf("전체 학습 오차(MSE) : %0.5f\n",round(tex1,4))
input("    학습이 완료되었습니다. 테스트 데이터로 실험을 해봅시다!")
    figure(2)
%%
load mnist\testingData.mat;
test = reshape(images,28^2,[]);
one_hot = diag(ones(1,max(labels)+1));
labels = one_hot(labels+1,:)';
y = labels';

results = [];
for i= 1 : 10000
    f1 = Relu(test(:,i)' * node_1w + node_1b');
    f2 = Relu(f1 * node_2w + node_2b');
    f3 = exp(f2 * node_3w + node_3b') / sum(exp(f2 * node_3w + node_3b'));
    results(i) =  min( y(i,:) == (f3 ==max(f3)));
end

fprintf("전체 테스트 데이터 학습 결과 %0.2f %% 정확도\n",round(mean(results)*100,2))
im=zeros(28,28,10);
results = [];
for i= 1 : 10
    f1 = Relu(test(:,i)' * node_1w + node_1b');
    f2 = Relu(f1 * node_2w + node_2b');
    f3 = exp(f2 * node_3w + node_3b') / sum(exp(f2 * node_3w + node_3b'));
    results(i) =  min( y(i,:) == (f3 ==max(f3)));
    
    im(:,:,i) = reshape(test(:,i)',28,28);
    title(find(f3 == max(f3))-1)
    input("")
end