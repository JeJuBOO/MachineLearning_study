function [out,pool_idx] = Pooling(in,num,stride)
arguments
    in (:,:,:,:) double
    num (1,1) double
    stride (1,1) double = 1
end
[col,row,ch,n] = size(in);
out = zeros(col/num,row/num,ch,n);
pool_idx = zeros(col/num,row/num,ch,n);
j_out = 0;i_out = 0;

for j = 1 :stride: col/num
    j_out = j_out+1;
    for i = 1 :stride: row/num
        i_out = i_out+1;
        [out(j_out,i_out,:,:),pool_idx(j_out,i_out,:,:)] = ...
            max(in((j-1)*num+1:num*j,(i-1)*num+1:num*i,:,:),[],[1,2],'linear');
    end
    i_out = 0;
end
end
