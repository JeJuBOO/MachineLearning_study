function out = Convolution(in,kernel)
arguments
    in (:,:,:,:) double
    kernel (:,:,:,:) double
end

[p_col,p_row,~,in_n] = size(in);
[k_col,k_row,k_ch,k_n] = size(kernel);
j = p_col+k_col-1;
i = p_row+k_row-1;
out = zeros(j,i,k_n,in_n);

for n = 1:in_n %전체 이미지 개수
    for k = 1:k_n %전체 커널 개수
        for d = 1:k_ch %전체 커널 개수
            out(:,:,k,n) = out(:,:,k,n) + conv2(in(:,:,d,n),kernel(:,:,d),'full');
        end
    end
end


end
