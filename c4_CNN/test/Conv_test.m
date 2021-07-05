% 21.07.05 
% correlation
% 
clc,clear,close all
%% 이미지 로드
% in = zeros(10); in(3:7,3:7) = ones(5);
in = double(imread('tired.jpg'));
[p_col,p_row,in_ch,in_n] = size(in);
kernel= [1 2 1
        0 0 0
        -1 -2 -1];
[k_col,k_row,~,k_n] = size(kernel);
kernel = reshape([kernel,kernel,kernel],[k_col,k_row,in_ch,k_n]);


[k_col,k_row,k_ch,k_n] = size(kernel);
j = length(1 :1: p_col - k_col+1);
i = length(1 :1: p_row - k_row+1);
out = zeros(j,i,k_n,in_n);
j_out = 0;i_out = 0;
s = 1;

tic
for n = 1:in_n %전체 이미지 개수
    for k = 1:k_n %전체 커널 개수
            
            for j = 1 :s: p_col - k_col+1
                j_out = j_out+1;
                for i = 1 :s: p_row - k_row+1
                    i_out = i_out+1;
                    out(j_out,i_out,k,n) = ...
                        sum(in(j:k_col-1+j,i:k_row-1+i,:,n).*kernel(:,:,:,k),[1,2,3]);
                end
                i_out = 0;
            end
            j_out = 0;

    end
end
time1 = toc;
figure()
subplot(1,2,1)
imshow(out)
txt = ['경과 시간은 ',num2str(time1)];
title("My code")
xlabel(txt)

Y1 = zeros([p_col-k_col+1,p_row-k_row+1,k_n,in_n]);
subplot(1,2,2)
tic
for n = 1:in_n %전체 이미지 개수
    for k = 1:k_n %전체 커널 개수
        for d = 1:k_ch %전체 커널 개수
            Y1(:,:,k,n) = Y1(:,:,k,n) + filter2(kernel(:,:,d),in(:,:,d,n),'valid');
        end
    end
end
time2 = toc;
imshow(Y1)
txt = ['경과 시간은 ',num2str(time2)];
title("matlab function")
xlabel(txt)
% Y3 = zeros([p_col-k_col+1,p_row-k_row+1,k_n,in_n]);
% figure()
% subplot(1,2,1)
% for n = 1:in_n %전체 이미지 개수
%     for k = 1:k_n %전체 커널 개수
%         for d = 1:k_ch %전체 커널 개수
%             Y3(:,:,k,n) = Y3(:,:,k,n) + conv2(in(:,:,d,n),kernel(:,:,d),'valid');
%         end
%     end
% end
% imshow(Y3)
% Y2 = zeros([p_col-k_col+1,p_row-k_row+1,k_n,in_n]);
% subplot(1,2,2)
% for n = 1:in_n %전체 이미지 개수
%     for k = 1:k_n %전체 커널 개수
%         for d = 1:k_ch %전체 커널 개수
%              kernel(:,:,d) = rot90(squeeze(kernel(:,:,d)),2);
%             Y2(:,:,k,n) = Y2(:,:,k,n) + filter2(kernel(:,:,d),in(:,:,d,n),'valid');
%         end
%     end
% end
% imshow(Y2)
