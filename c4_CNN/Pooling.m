function out = Pooling(in,num,stride)
arguments
    in (:,:,:,:) double
    num (1,1) double
    stride (1,1) double = 1
end
[col,row,~,n] = size(in);
j = length(1 :stride: col - num+1);
i = length(1 :stride: row - num+1);
out = zeros(j,i,1,n);
j_out = 0;i_out = 0;

for j = 1 :stride: col - num+1
    j_out = j_out+1;
    for i = 1 :stride: row - num+1
        i_out = i_out+1;
        out(j_out,i_out,1,:) = ...
            max(in(j:num-1+j,i:num-1+i,:,:),[],[1,2,3]);
    end
    i_out = 0;
end
end
